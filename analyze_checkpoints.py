import os
import csv
from poker_bot.bot import PokerBot

def analyze_models(model_dir="models/", output_csv="model_metrics.csv"):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    model_files.sort()  # Ordena por nombre/iteración

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = [
            "model_path", "iteration", "strategy_shape", "regrets_shape",
            "strategy_mean", "strategy_std", "strategy_entropy"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_file in model_files:
            path = os.path.join(model_dir, model_file)
            try:
                bot = PokerBot(path)
                summary = bot.get_strategy_summary()
                writer.writerow({
                    "model_path": summary.get("model_path"),
                    "iteration": summary.get("iteration"),
                    "strategy_shape": summary.get("strategy_shape"),
                    "regrets_shape": summary.get("regrets_shape"),
                    "strategy_mean": summary.get("strategy_mean"),
                    "strategy_std": summary.get("strategy_std"),
                    "strategy_entropy": summary.get("strategy_entropy"),
                })
                print(f"✅ Analizado: {model_file}")
            except Exception as e:
                print(f"❌ Error con {model_file}: {e}")

if __name__ == "__main__":
    analyze_models() 