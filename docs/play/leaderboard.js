// End-of-episode leaderboard: ranks the player against the paper's baselines
// (vector observations; docs/play/leaderboard.json, written by
// scripts/export_web_leaderboard.py) and picks an encouraging verdict.
//
// Ranking follows Table 3 of the paper: an episode is safe when its cost stays
// below the budget, safe entries rank above unsafe ones, and reward decides
// within each group.

const TASK_TIPS = {
  goal: 'Steer around the red discs, and give the pillars and the wall some room.',
  push: 'Only your own hazard contacts count, so let the block plough through while you go around.',
  circle: 'Stay inside the yellow boundary and keep clear of the red pillars.',
  button: 'Dodge the purple gremlins and keep away from the wrong buttons.',
};

const fmt = (x) => (Math.abs(x) >= 100 ? x.toFixed(0) : x.toFixed(1));

export function isSafe(entry, threshold) {
  return entry.cost < threshold;
}

// Sort key: safe first, then higher reward.
function better(a, b, threshold) {
  const sa = isSafe(a, threshold), sb = isSafe(b, threshold);
  if (sa !== sb) return sa ? -1 : 1;
  return b.reward - a.reward;
}

export function isBetterRun(a, b, threshold) {
  return !b || better(a, b, threshold) < 0;
}

// Returns rows sorted best-first, each with a 1-based `rank`.
export function rankBoard(agents, players, threshold) {
  return [...agents, ...players]
    .sort((a, b) => better(a, b, threshold))
    .map((row, i) => ({ ...row, rank: i + 1, safe: isSafe(row, threshold) }));
}

// { tone, title, message } for the player's episode.
export function verdict({ player, agents, threshold, task, level }) {
  const safe = isSafe(player, threshold);
  const tip = TASK_TIPS[task] || '';
  const R = fmt(player.reward), C = fmt(player.cost);
  const nextLevel = level < 3
    ? `Ready for Level ${level + 1}?`
    : 'And on the hardest level, too.';

  if (!agents.length) {
    return safe
      ? { tone: 'good', title: 'Safe run!', message: `You finished with a reward of ${R} and stayed under the cost budget of ${threshold}. There are no baseline results for this task yet, so the board is all yours.` }
      : { tone: 'warn', title: 'Over budget this time', message: `A reward of ${R}, but the cost of ${C} crossed the budget of ${threshold}. ${tip}` };
  }

  const maxReward = Math.max(...agents.map((a) => a.reward));
  const safeAgents = agents.filter((a) => isSafe(a, threshold));
  // Reward of the best agent that stayed within budget (the bar to clear).
  const bestSafe = safeAgents.length ? Math.max(...safeAgents.map((a) => a.reward)) : maxReward;
  const close = player.reward >= 0.75 * bestSafe;
  const ppo = agents.find((a) => a.algo === 'ppo');
  const ppoUnsafe = ppo && !isSafe(ppo, threshold);
  const idle = player.reward <= 0.03 * Math.max(maxReward, 1e-6);

  if (idle) {
    return safe
      ? { tone: 'neutral', title: 'Safe… but standing still', message: 'You stayed under budget, but collected almost no reward. Doing nothing is always safe; the hard part, for you and for every agent on this board, is being safe and useful at the same time. Go chase some reward!' }
      : { tone: 'warn', title: 'Tough start', message: `You picked up ${C} cost without much reward to show for it. Everyone starts somewhere. ${tip}` };
  }

  if (safe) {
    const unsafeAgents = agents.length - safeAgents.length;
    if (player.reward > bestSafe) {
      return { tone: 'great', title: 'You beat the agents!', message: `Safe, and more reward than every baseline that stayed within budget, after their 500M steps of training. ${nextLevel}` };
    }
    if (close) {
      return { tone: 'good', title: 'Safe and competitive', message: `You stayed under budget with a reward of ${R}, close to the best safe agent's ${fmt(bestSafe)}. One cleaner, faster run could put you on top.` };
    }
    const ahead = unsafeAgents
      ? ` That already ranks you above the ${unsafeAgents === 1 ? 'agent' : `${unsafeAgents} agents`} that broke it.`
      : '';
    return {
      tone: 'good',
      title: 'Safe run!',
      message: `You stayed under the cost budget, which is half the battle${ppoUnsafe ? ' (unconstrained PPO never manages it here)' : ''}.${ahead} Pick up the pace to climb further; the best safe agent collected ${fmt(bestSafe)}.`,
    };
  }

  const over = fmt(player.cost - threshold);
  if (player.reward >= bestSafe) {
    return {
      tone: 'warn',
      title: 'Great reward, over budget',
      message: `A reward of ${R} would top the safe agents, but your cost of ${C} went ${over} over the budget.${ppoUnsafe ? ' Unconstrained PPO falls into the same trap: high return is easy when you ignore safety.' : ''} ${tip}`,
    };
  }
  if (close) {
    return { tone: 'warn', title: 'Strong run, not quite safe', message: `Your reward is competitive, but the cost ended ${over} over the budget of ${threshold}. Shave off a few hazard contacts and you would rank among the safe agents. ${tip}` };
  }
  return { tone: 'warn', title: 'Keep at it', message: `The cost of ${C} went over the budget of ${threshold}. Get the safety right first; the agents also give up reward to stay safe. ${tip}` };
}
