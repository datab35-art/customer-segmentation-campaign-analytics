"""Generate synthetic member profiles and campaign data for segmentation analysis."""

import random

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

NUM_MEMBERS = 3000
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
PLAN_TYPES = ["HMO", "PPO", "EPO", "HDHP"]
CHANNELS = ["email", "sms", "mail", "phone"]
CAMPAIGN_TYPES = ["wellness", "preventive_care", "plan_renewal", "cost_savings", "digital_onboard"]


def generate_member_profiles():
    members = []
    for i in range(1, NUM_MEMBERS + 1):
        age = random.randint(18, 80)
        region = random.choice(REGIONS)
        plan_type = random.choices(PLAN_TYPES, weights=[0.35, 0.30, 0.15, 0.20])[0]
        tenure_months = random.randint(1, 120)

        # utilization features
        annual_claims = max(0, int(np.random.poisson(3 + age / 20)))
        er_visits = max(0, int(np.random.poisson(0.3 if age > 50 else 0.1)))
        rx_fills = max(0, int(np.random.poisson(2 + age / 25)))
        preventive_visits = max(0, int(np.random.poisson(1.2)))

        # engagement features
        portal_logins = max(0, int(np.random.poisson(6 if age < 45 else 3)))
        mobile_sessions = max(0, int(np.random.poisson(10 if age < 35 else 3)))
        calls_to_service = max(0, int(np.random.poisson(2)))
        email_opens_last_6m = max(0, int(np.random.poisson(4)))

        # cost features
        total_cost = max(0, np.random.lognormal(mean=7.5 + annual_claims * 0.1, sigma=0.8))
        member_cost_share = total_cost * np.random.uniform(0.1, 0.4)

        # satisfaction
        satisfaction = np.clip(np.random.normal(7.0 - calls_to_service * 0.3, 1.5), 1, 10)

        members.append({
            "member_id": f"MBR-{i:05d}",
            "age": age,
            "region": region,
            "plan_type": plan_type,
            "tenure_months": tenure_months,
            "annual_claims": annual_claims,
            "er_visits": er_visits,
            "rx_fills": rx_fills,
            "preventive_visits": preventive_visits,
            "portal_logins_6m": portal_logins,
            "mobile_sessions_6m": mobile_sessions,
            "calls_to_service_6m": calls_to_service,
            "email_opens_6m": email_opens_last_6m,
            "total_cost_ytd": round(total_cost, 2),
            "member_cost_share_ytd": round(member_cost_share, 2),
            "satisfaction_score": round(satisfaction, 1),
        })

    return pd.DataFrame(members)


def generate_campaign_data(members_df):
    campaigns = []
    campaign_id = 0

    for campaign_type in CAMPAIGN_TYPES:
        sampled = members_df.sample(frac=np.random.uniform(0.4, 0.7), random_state=campaign_id)

        for _, member in sampled.iterrows():
            campaign_id += 1
            channel = random.choices(CHANNELS, weights=[0.40, 0.25, 0.20, 0.15])[0]

            # response rates vary by age and engagement
            base_open = 0.35 if channel == "email" else 0.50 if channel == "sms" else 0.20
            engagement_boost = min(0.15, member["portal_logins_6m"] * 0.01 + member["email_opens_6m"] * 0.02)
            open_rate = min(0.9, base_open + engagement_boost + np.random.normal(0, 0.05))
            opened = random.random() < open_rate

            clicked = False
            converted = False
            if opened:
                click_prob = 0.25 + engagement_boost
                clicked = random.random() < click_prob
                if clicked:
                    convert_prob = 0.15 + (0.05 if member["satisfaction_score"] > 7 else 0)
                    converted = random.random() < convert_prob

            campaigns.append({
                "campaign_id": f"CMP-{campaign_id:06d}",
                "member_id": member["member_id"],
                "campaign_type": campaign_type,
                "channel": channel,
                "opened": opened,
                "clicked": clicked,
                "converted": converted,
                "cost": round(np.random.uniform(0.50, 3.00) if channel != "phone" else np.random.uniform(5.0, 15.0), 2),
            })

    return pd.DataFrame(campaigns)


if __name__ == "__main__":
    print("Generating member profiles...")
    members = generate_member_profiles()
    members.to_csv("data/member_profiles.csv", index=False)

    print("Generating campaign data...")
    campaigns = generate_campaign_data(members)
    campaigns.to_csv("data/campaign_results.csv", index=False)

    print(f"\nMembers: {len(members):,}")
    print(f"Campaign touches: {len(campaigns):,}")
    print(f"Campaign types: {campaigns['campaign_type'].nunique()}")
    print(f"Avg open rate: {campaigns['opened'].mean():.1%}")
    print(f"Avg click rate: {campaigns['clicked'].mean():.1%}")
    print(f"Avg conversion rate: {campaigns['converted'].mean():.1%}")
    print("Files saved to data/")
