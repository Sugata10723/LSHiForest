import pandas as pd
from sklearn.metrics import accuracy_score

class AttackAnalyzer:
    @staticmethod
    def analyze_by_attack(y_true, y_pred, attack_cats):
        results = {}
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'attack_cat': attack_cats})
        
        for attack_cat in df['attack_cat'].unique():
            if attack_cat == 'normal':
                continue
            
            subset = df[df['attack_cat'] == attack_cat]
            accuracy = accuracy_score(subset['y_true'], subset['y_pred'])
            results[attack_cat] = {
                'accuracy': accuracy,
                'count': len(subset)
            }
            
        return results

    @staticmethod
    def print_analysis(results):
        print("\n--- Analysis by Attack Method ---")
        for attack, result in results.items():
            print(f"  Attack: {attack}")
            print(f"    Accuracy: {result['accuracy']:.4f}")
            print(f"    Count: {result['count']}")
        print("---------------------------------")
