import numpy as np
import pandas as pd

# -------------------------------
# Global configuration
# -------------------------------
RANDOM_SEED = 42
N_CUSTOMERS = 5000
N_WEEKS = 52

np.random.seed(RANDOM_SEED)

# -------------------------------
# Customer-level data
# -------------------------------
def generate_customers():
    customers = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(N_CUSTOMERS)],
        "plan": np.random.choice(
            ["Basic", "Pro", "Enterprise"],
            size=N_CUSTOMERS,
            p=[0.55, 0.30, 0.15]
        )
    })

    mrr_ranges = {
        "Basic": (30, 80),
        "Pro": (120, 300),
        "Enterprise": (800, 2000)
    }

    customers["MRR"] = customers["plan"].apply(
        lambda p: np.random.uniform(*mrr_ranges[p])
    )

    customers["CLV"] = customers["MRR"] * np.random.uniform(8, 24)

    customers["retention_cost"] = customers["plan"].map({
        "Basic": np.random.uniform(20, 60),
        "Pro": np.random.uniform(80, 200),
        "Enterprise": np.random.uniform(300, 800)
    })

    return customers


# -------------------------------
# Weekly behavioral data
# -------------------------------
def generate_behavior(customers):
    records = []

    for _, row in customers.iterrows():
        base_usage = {
            "Basic": np.random.uniform(5, 15),
            "Pro": np.random.uniform(15, 40),
            "Enterprise": np.random.uniform(40, 90)
        }[row.plan]

        engagement_velocity = np.random.normal(0, 0.3)
        friction_rate = np.random.uniform(0.1, 0.5)

        usage = base_usage

        for week in range(N_WEEKS):
            usage = max(0, usage + engagement_velocity + np.random.normal(0, 1))
            support_tickets = np.random.poisson(friction_rate)

            if support_tickets >= 3:
                engagement_velocity -= 0.4  # friction accelerates decline

            records.append({
                "customer_id": row.customer_id,
                "week": week,
                "weekly_usage": usage,
                "support_tickets": support_tickets
            })

    return pd.DataFrame(records)


# -------------------------------
# Churn labeling (behavior-based)
# -------------------------------
def label_churn(customers, behavior):
    recent_usage = (
        behavior.groupby("customer_id")
        .tail(8)
        .groupby("customer_id")["weekly_usage"]
        .mean()
    )

    customers["churned"] = customers["customer_id"].map(
        recent_usage < 2.0
    ).fillna(False)

    return customers


# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    customers = generate_customers()
    behavior = generate_behavior(customers)
    customers = label_churn(customers, behavior)

    customers.to_csv("customers.csv", index=False)
    behavior.to_csv("behavior.csv", index=False)

    print("Synthetic data generated:")
    print(customers.head())
