# ============================================================================
# REWARD ANALYSIS CELL
# ============================================================================
# INSTRUCTIONS: 
# 1. Copy this ENTIRE cell into your Colab notebook
# 2. Place it AFTER the "Rollout Trained Policy" cell (where you run the trained policy)
# 3. Run it AFTER you have a 'rollout' variable from the policy rollout
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt

def analyze_rewards(rollout, reward_config):
    """
    Analyze reward distribution from a training rollout.
    Shows which rewards dominate, which are negligible, and tuning suggestions.
    """
    
    # Extract rewards from rollout
    # The rollout should have rewards stored in state.info['rewards']
    rewards_data = {}
    
    # Try to get rewards from the rollout
    try:
        # Collect rewards across all timesteps
        for state in rollout:
            if hasattr(state, 'info') and 'rewards' in state.info:
                for name, value in state.info['rewards'].items():
                    if name not in rewards_data:
                        rewards_data[name] = []
                    rewards_data[name].append(float(value))
    except Exception as e:
        print(f"Error extracting rewards: {e}")
        print("Make sure 'rollout' contains state objects with info['rewards']")
        return
    
    if not rewards_data:
        print("No rewards found in rollout. Trying alternative extraction...")
        # Alternative: if rollout is just states without rewards, we need to re-run
        print("Please ensure you ran the rollout with reward tracking enabled.")
        return
    
    # Calculate statistics
    print("\n" + "="*80)
    print("                         REWARD ANALYSIS REPORT")
    print("="*80)
    
    stats = {}
    for name, values in rewards_data.items():
        arr = np.array(values)
        stats[name] = {
            'mean': np.mean(arr),
            'std': np.std(arr),
            'min': np.min(arr),
            'max': np.max(arr),
            'sum': np.sum(arr),
            'abs_mean': np.mean(np.abs(arr))
        }
    
    # Sort by absolute contribution (most impactful first)
    sorted_rewards = sorted(stats.items(), key=lambda x: abs(x[1]['sum']), reverse=True)
    
    # Print table
    print(f"\n{'Reward Name':<30} | {'Mean':>10} | {'Sum':>12} | {'Min':>10} | {'Max':>10}")
    print("-"*80)
    
    total_positive = 0
    total_negative = 0
    
    for name, s in sorted_rewards:
        mean_str = f"{s['mean']:.4f}"
        sum_str = f"{s['sum']:.2f}"
        min_str = f"{s['min']:.4f}"
        max_str = f"{s['max']:.4f}"
        print(f"{name:<30} | {mean_str:>10} | {sum_str:>12} | {min_str:>10} | {max_str:>10}")
        
        if s['sum'] > 0:
            total_positive += s['sum']
        else:
            total_negative += s['sum']
    
    print("-"*80)
    print(f"{'TOTAL POSITIVE':<30} | {'':<10} | {total_positive:>12.2f}")
    print(f"{'TOTAL NEGATIVE':<30} | {'':<10} | {total_negative:>12.2f}")
    print(f"{'NET REWARD':<30} | {'':<10} | {total_positive + total_negative:>12.2f}")
    
    # Recommendations
    print("\n" + "="*80)
    print("                         TUNING RECOMMENDATIONS")
    print("="*80)
    
    # Find dominant rewards
    if sorted_rewards:
        top_positive = [(n, s) for n, s in sorted_rewards if s['sum'] > 0][:3]
        top_negative = [(n, s) for n, s in sorted_rewards if s['sum'] < 0][:3]
        
        print("\n🟢 TOP POSITIVE REWARDS (driving behavior):")
        for name, s in top_positive:
            pct = (s['sum'] / total_positive * 100) if total_positive > 0 else 0
            print(f"   {name}: {s['sum']:.2f} ({pct:.1f}% of positive rewards)")
        
        print("\n🔴 TOP NEGATIVE REWARDS (penalizing behavior):")
        for name, s in top_negative:
            pct = (s['sum'] / abs(total_negative) * 100) if total_negative < 0 else 0
            print(f"   {name}: {s['sum']:.2f} ({pct:.1f}% of penalties)")
        
        # Zero-impact rewards
        negligible = [(n, s) for n, s in sorted_rewards if abs(s['sum']) < 1.0]
        if negligible:
            print("\n⚪ NEGLIGIBLE REWARDS (consider removing or boosting):")
            for name, s in negligible[:5]:
                print(f"   {name}: {s['sum']:.4f}")
        
        # Spinning detection
        ang_vel_reward = stats.get('ang_vel_xy', {}).get('sum', 0)
        lin_vel_reward = stats.get('tracking_lin_vel', {}).get('sum', 0)
        
        if abs(ang_vel_reward) > abs(lin_vel_reward):
            print("\n⚠️  WARNING: Angular velocity penalty > Linear velocity reward")
            print("   Robot might be spinning! Consider:")
            print("   - Increase tracking_lin_vel scale")
            print("   - Check if wheels can achieve commanded velocities")
    
    # Plot
    plt.figure(figsize=(14, 6))
    
    # Bar chart of mean rewards
    names = [n for n, _ in sorted_rewards]
    means = [s['mean'] for _, s in sorted_rewards]
    colors = ['green' if m > 0 else 'red' for m in means]
    
    plt.barh(names, means, color=colors, alpha=0.7)
    plt.xlabel('Mean Reward per Step')
    plt.title('Reward Contribution Analysis')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    
    return stats


# ============================================================================
# RUN THE ANALYSIS
# ============================================================================
# Make sure you have 'rollout' from the policy rollout cell
# and CONFIG from your config setup

try:
    stats = analyze_rewards(rollout, CONFIG.reward)
except NameError as e:
    print(f"Error: {e}")
    print("\nMake sure you have:")
    print("1. Run the 'Rollout Trained Policy' cell first")
    print("2. The 'rollout' variable contains the episode data")
