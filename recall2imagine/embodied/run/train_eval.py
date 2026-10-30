import re

import embodied
import numpy as np
from tqdm.auto import tqdm


def train_eval(
    agent, train_env, eval_env, train_replay, eval_replay, logger, args, config):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
  should_sync = embodied.when.Every(args.sync_every)
  step = logger.step
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  action_budget = config.task.startswith('atari_')
  eval_scores = []
  last_eval_step = None
  print('Observation space:', embodied.format(train_env.obs_space), sep='\n')
  print('Action space:', embodied.format(train_env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', train_env, ['step'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])

  nonzeros = set()
  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    if mode == 'eval':
      eval_scores.append(score)
    logger.add({
        'length': length, 'score': score,
        'reward_rate': (ep['reward'] - ep['reward'].min() >= 0.1).mean(),
    }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
    print(f'Episode has {length} steps and return {score:.1f}.')
    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix=f'{mode}_stats')

  driver_train = embodied.Driver(train_env)
  driver_train.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
  # Atari100k counts agent actions, including prefill, but not reset records.
  driver_train.on_step(lambda tran, _: step.increment()
                       if not action_budget or not tran['is_first'] else None)
  driver_train.on_step(train_replay.add)
  driver_eval = embodied.Driver(eval_env)
  driver_eval.on_step(eval_replay.add)
  driver_eval.on_episode(lambda ep, worker: per_episode(ep, mode='eval'))

  train_replay.set_agent(agent)
  train_replay.maybe_restore()
  random_agent = embodied.RandomAgent(train_env.act_space, seed=config.seed)
  print('Prefill train dataset.')
  while len(train_replay) < max(args.batch_steps, args.train_fill):
    driver_train(random_agent.policy, steps=100)
  print('Prefill eval dataset.')
  while len(eval_replay) < max(args.batch_steps, args.eval_fill):
    driver_eval(random_agent.policy, steps=100)
  logger.add(metrics.result())
  logger.write()

  if config.replay == 'lfs':
    dataset_train = agent.dataset(train_replay, shared_memory=True)
  else:
    dataset_train = agent.dataset(train_replay.dataset, shared_memory=False)
  dataset_eval = agent.dataset(eval_replay.dataset, shared_memory=False)
  state = [None]  # To be writable from train step function below.
  batch = [None]
  def train_step(tran, worker):
    if action_budget and tran['is_first']:
      return
    for _ in range(should_train(step)):
      with timer.scope('dataset_train'):
        batch[0] = next(dataset_train)
      outs, state[0], mets = agent.train(batch[0], state[0])
      metrics.add(mets, prefix='train')
      if 'priority' in outs:
        train_replay.prioritize(outs['key'], outs['priority'])
      updates.increment()
    if should_sync(updates):
      agent.sync()
    if should_log(step):
      logger.add(metrics.result())
      logger.add(agent.report(batch[0]), prefix='report')
      with timer.scope('dataset_eval'):
        eval_batch = next(dataset_eval)
      logger.add(agent.report(eval_batch), prefix='eval')
      logger.add(train_replay.stats, prefix='replay')
      logger.add(eval_replay.stats, prefix='eval_replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  driver_train.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt', parallel=False)
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  restore_keys = ['train_replay', 'eval_replay', 'agent', 'step']
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint, keys=restore_keys)
  checkpoint.load_or_save(keys=restore_keys)
  if hasattr(agent, '_updates'):
    updates.load(int(agent._updates))
  should_save(step)  # Register that we jused saved.

  print('Start training loop.')
  policy_train = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  policy_eval = lambda *args: agent.policy(*args, mode='eval')
  def evaluate():
    nonlocal last_eval_step
    agent.sync()
    eval_scores.clear()
    print('Starting evaluation at step', int(step))
    driver_eval.reset()
    driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps))
    logger.add({'score_mean': float(np.mean(eval_scores)),
                'score_std': float(np.std(eval_scores)),
                'episodes': len(eval_scores)}, prefix='eval')
    logger.write()
    last_eval_step = int(step)

  with tqdm(total=int(args.steps), initial=int(step), desc='CT-WM Atari' if action_budget else 'CT-WM',
            unit='actions' if action_budget else 'records') as progress:
    while step < args.steps:
      if should_eval(step):
        evaluate()
      before = int(step)
      driver_train(policy_train, steps=min(100, int(args.steps) - before))
      progress.update(int(step) - before)
      progress.set_postfix(updates=int(updates), replay=len(train_replay), refresh=False)
      if should_save(step):
        checkpoint.save()
  # Save the exact training endpoint even when no wall-clock save is due.
  checkpoint.save()
  logger.add(metrics.result())
  logger.add({'updates': int(updates)}, prefix='run')
  if last_eval_step != int(step):
    evaluate()
  logger.write()
