from ats_scorer import ATSScorer
def calculate_optimization_gain(resume_text, jd_text):
    scorer = ATSScorer()

    # 1. Get Initial Score
    print("--- 1. Calculating Initial Score ---")
    initial_result = scorer.calculate_ats_score(resume_text, jd_text)
    initial_score = initial_result['final_score']
    suggestions = initial_result['missing_keywords']

    print(f"Initial Score: {initial_score}")
    print(f"Missing Keywords Found: {suggestions}")

    # 2. Simulate User Improving Resume (Injecting Keywords)
    # We append the suggested keywords to the resume to simulate optimization
    optimized_resume_text = resume_text + " " + " ".join(suggestions)

    # 3. Get New Score
    print("\n--- 2. Calculating Optimized Score ---")
    optimized_result = scorer.calculate_ats_score(optimized_resume_text, jd_text)
    new_score = optimized_result['final_score']
    print(f"New Score: {new_score}")

    # 4. Calculate Gain %
    if initial_score == 0:
        gain = 0
    else:
        gain = ((new_score - initial_score) / initial_score) * 100

    return gain


# --- TEST DATA ---
# Replace these with real text to test
dummy_resume = """
I am a python developer with experience in flask and web development.
I know how to code and deploy applications.
"""

dummy_jd = """
Looking for a Python Developer with experience in Flask, Docker, Kubernetes, 
and Cloud computing. Must know SQL and API development.
"""

# Run Calculation
gain_percent = calculate_optimization_gain(dummy_resume, dummy_jd)

print("\n" + "=" * 30)
print(f"🚀 ATS Optimization Gain: {gain_percent:.2f}%")
print("=" * 30)