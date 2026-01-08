"""
AI Campaign Consultant Logic
============================
Integrates Generative AI to provide strategic advice based on causal model outputs.
"""
import logging
import random

logger = logging.getLogger(__name__)

class AIConsultant:
    def __init__(self, api_key=None, provider="openai"):
        self.api_key = api_key
        self.provider = provider
        
    def generate_advice(self, campaign_data, prediction_data, counterfactual_data):
        """
        Generate strategic advice based on campaign metrics.
        """
        if self.api_key:
            return self._call_llm(campaign_data, prediction_data, counterfactual_data)
        else:
            return self._generate_synthetic_advice(campaign_data, prediction_data, counterfactual_data)

    def _generate_synthetic_advice(self, c, p, cf):
        """Generate high-quality rule-based advice (Fallback)."""
        
        # 1. Analyze Status
        ratio = p['funding_ratio']
        if ratio >= 1.2:
            sentiment = "enthusiastic"
            headline = "🚀 You are positioned for a breakout success!"
        elif ratio >= 1.0:
            sentiment = "positive"
            headline = "✅ You are on track, but stay vigilant."
        elif ratio >= 0.7:
            sentiment = "cautious"
            headline = "⚠️ Success is possible, but you need adjustments."
        else:
            sentiment = "critical"
            headline = "🛑 Significant risks detected. Pivot recommended."

        # 2. Price Analysis
        price_diff = cf['optimal_price'] - c['avg_reward_price']
        if abs(price_diff) < 5:
            price_advice = "Your pricing is scientifically optimal. Don't touch it."
        elif price_diff > 0:
            price_advice = f"You are undercharging! Our causal model suggests raising the price to **${cf['optimal_price']}**. Your backers view this category as premium."
        else:
            price_advice = f"Your price is too high. Lowering it to **${cf['optimal_price']}** could unlock significant demand based on elasticity data."

        # 3. Strategy
        strategies = []
        if c['campaign_duration'] > 45:
            strategies.append("Your campaign is too long (momentum killer). Compress it to 30-35 days to create urgency.")
        
        if p['confidence_high'] - p['confidence_low'] > 20000:
            strategies.append("High uncertainty detected. Secure 30% of funding from friends/family on Day 1 to anchor social proof.")
            
        if c['goal_ambition'] > 1.5:
            strategies.append(f"Your goal is {c['goal_ambition']:.1f}x the category median. Consider splitting this into 'Phase 1' (lower goal) and 'Phase 2' stretch goals.")

        if not strategies:
            strategies.append("Focus on community building 2 weeks before launch.")

        return {
            "headline": headline,
            "summary": f"Our causal AI estimates a **{p['success_probability']*100:.0f}% chance** of success. {price_advice}",
            "strategies": strategies,
            "risk_score": max(1, 10 - int(ratio * 10)),
            "tone": sentiment
        }

    def _call_llm(self, c, p, cf):
        """Placeholder for real LLM call."""
        advice = self._generate_synthetic_advice(c, p, cf)
        advice['headline'] = "[AI GENERATED] " + advice['headline']
        return advice
