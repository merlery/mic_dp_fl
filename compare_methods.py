"""
Comparison script for baseline (IN) vs MIC-based transformation methods
"""
import subprocess
import sys
import os
import pandas as pd
import argparse
from datetime import datetime


def run_experiment(data, nclient, nclass, ncpc, model, mode, round, epsilon, sr, lr, flr, physical_bs, E, experiment_name):
    """Run a single experiment and return the result"""
    print(f"\n{'='*60}")
    print(f"Running: {experiment_name}")
    print(f"Model: {model}, Mode: {mode}, Epsilon: {epsilon}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable, 'FedAverage.py',
        '--data', data,
        '--nclient', str(nclient),
        '--nclass', str(nclass),
        '--ncpc', str(ncpc),
        '--model', model,
        '--mode', mode,
        '--round', str(round),
        '--epsilon', str(epsilon),
        '--sr', str(sr),
        '--lr', str(lr),
        '--flr', str(flr),
        '--physical_bs', str(physical_bs),
        '--E', str(E)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse accuracy from output
        output_lines = result.stdout.split('\n')
        best_acc = None
        for line in output_lines:
            if 'Best accuracy:' in line:
                best_acc = float(line.split(':')[1].strip())
                break
        
        return {
            'experiment': experiment_name,
            'model': model,
            'mode': mode,
            'epsilon': epsilon,
            'accuracy': best_acc,
            'status': 'success'
        }
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        print(f"Error output: {e.stderr}")
        return {
            'experiment': experiment_name,
            'model': model,
            'mode': mode,
            'epsilon': epsilon,
            'accuracy': None,
            'status': 'failed',
            'error': str(e)
        }


def compare_methods(data='mnist', nclient=100, nclass=10, ncpc=2, mode='CDP', 
                   round=60, epsilon=2, sr=1, lr=5e-3, flr=1e-2, physical_bs=64, E=1):
    """
    Compare baseline (IN) and MIC-based transformation methods
    """
    results = []
    
    # Model mapping: baseline -> MIC
    model_mapping = {
        'mnist_fully_connected_IN': 'mnist_fully_connected_MIC',
        'resnet18_IN': 'resnet18_MIC',
        'alexnet_IN': 'alexnet_MIC',
        'purchase_fully_connected_IN': 'purchase_fully_connected_MIC'
    }
    
    # Determine which model to use based on data
    if data == 'mnist':
        baseline_model = 'mnist_fully_connected_IN'
        mic_model = 'mnist_fully_connected_MIC'
    elif data == 'cifar10':
        baseline_model = 'resnet18_IN'
        mic_model = 'resnet18_MIC'
    elif data == 'purchase':
        baseline_model = 'purchase_fully_connected_IN'
        mic_model = 'purchase_fully_connected_MIC'
    else:
        print(f"Warning: No model mapping for {data}, using mnist models")
        baseline_model = 'mnist_fully_connected_IN'
        mic_model = 'mnist_fully_connected_MIC'
    
    # Run baseline (IN) method
    baseline_result = run_experiment(
        data, nclient, nclass, ncpc, baseline_model, mode, round, 
        epsilon, sr, lr, flr, physical_bs, E, 
        f"Baseline ({baseline_model})"
    )
    results.append(baseline_result)
    
    # Run MIC method
    mic_result = run_experiment(
        data, nclient, nclass, ncpc, mic_model, mode, round, 
        epsilon, sr, lr, flr, physical_bs, E, 
        f"MIC-based ({mic_model})"
    )
    results.append(mic_result)
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'log/comparison_{data}_{mode}_eps{epsilon}_{timestamp}.csv'
    os.makedirs('log', exist_ok=True)
    df.to_csv(output_file, index=False)
    
    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\nResults saved to: {output_file}")
    
    # Calculate improvement
    if baseline_result['accuracy'] is not None and mic_result['accuracy'] is not None:
        improvement = mic_result['accuracy'] - baseline_result['accuracy']
        improvement_pct = (improvement / baseline_result['accuracy']) * 100
        print(f"\nAccuracy Improvement: {improvement:.4f} ({improvement_pct:+.2f}%)")
        if improvement > 0:
            print("✓ MIC method performs better!")
        elif improvement < 0:
            print("✗ Baseline method performs better")
        else:
            print("= Both methods perform similarly")
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare baseline (IN) vs MIC transformation methods')
    parser.add_argument('--data', type=str, default='mnist',
                        choices=['mnist', 'cifar10', 'cifar100', 'fashionmnist', 'emnist', 'purchase', 'chmnist'])
    parser.add_argument('--nclient', type=int, default=100)
    parser.add_argument('--nclass', type=int, default=10)
    parser.add_argument('--ncpc', type=int, default=2)
    parser.add_argument('--mode', type=str, default='CDP', choices=['CDP', 'LDP'])
    parser.add_argument('--round', type=int, default=60)
    parser.add_argument('--epsilon', type=int, default=2)
    parser.add_argument('--sr', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--flr', type=float, default=1e-2)
    parser.add_argument('--physical_bs', type=int, default=64)
    parser.add_argument('--E', type=int, default=1)
    
    args = parser.parse_args()
    
    compare_methods(
        data=args.data,
        nclient=args.nclient,
        nclass=args.nclass,
        ncpc=args.ncpc,
        mode=args.mode,
        round=args.round,
        epsilon=args.epsilon,
        sr=args.sr,
        lr=args.lr,
        flr=args.flr,
        physical_bs=args.physical_bs,
        E=args.E
    )

