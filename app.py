from flask import Flask, render_template, request, jsonify
import os
from file_parser import extract_text_from_file, clean_text
from ats_scorer import ATSScorer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ats-secret-123'

# Initialize ATS Scorer
ats_scorer = ATSScorer()


@app.route('/')
def user_page():
    return render_template('user.html')


@app.route('/analyze', methods=['POST'])
def analyze_resume():
    try:
        resume_file = request.files['resume']
        jd_file = request.files['jd']

        resume_text = clean_text(extract_text_from_file(resume_file))
        jd_text = clean_text(extract_text_from_file(jd_file))

        if not resume_text:
            return jsonify({'success': False, 'error': 'Resume text extract nahi hua'})
        if not jd_text:
            return jsonify({'success': False, 'error': 'JD text extract nahi hua'})

        result = ats_scorer.calculate_ats_score(resume_text, jd_text)

        return jsonify({
            'success': True,
            'score': result['final_score'],
            'suggestions': result['missing_keywords'],
            'breakdown': result['breakdown']
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/hr')
def hr_page():
    return render_template('hr.html')


@app.route('/hr/analyze', methods=['POST'])
def hr_analyze():
    try:
        jd_file = request.files['jd']
        resume_files = request.files.getlist('resumes')

        jd_text = clean_text(extract_text_from_file(jd_file))

        if not jd_text:
            return jsonify({'success': False, 'error': 'text not extracted'})

        results = []
        for resume_file in resume_files:
            resume_text = clean_text(extract_text_from_file(resume_file))

            if resume_text:
                result = ats_scorer.calculate_ats_score(resume_text, jd_text)

                results.append({
                    'name': resume_file.filename,
                    'score': result['final_score'],
                    'suggestions': result['missing_keywords'][:3]
                })

        results.sort(key=lambda x: x['score'], reverse=True)

        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("🚀 ATS System Started!")
    print("📍 User: http://localhost:5000")
    print("📍 HR: http://localhost:5000/hr")
    app.run(debug=False, host='0.0.0.0', port=5000)