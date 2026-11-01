# AI-Powered Phishing Detection System — SOF106 Group Project

A 6-person team project for the Principles of Artificial Intelligence course, building a phishing detector that combines text and image-based classification.

## My role: OCR Developer

Implemented the optical character recognition (OCR) module that extracts and cleans text from uploaded screenshots (e.g. suspicious email or SMS screenshots), feeding that text into the team's classification pipeline.

## How the system works

- **Text input:** messages are vectorized and classified using a Naive Bayes model trained on a labelled phishing/legitimate message dataset
- **Image input:** screenshots are passed through my OCR module to extract text, which is then cleaned and classified the same way as direct text input
- The system outputs a phishing likelihood assessment based on the trained model's prediction

## Team roles

| Role | Responsibility |
|---|---|
| Data Engineer | Dataset acquisition, cleaning, and TF-IDF vectorization |
| Text Model Developer | Built and trained the Naive Bayes phishing detection model |
| **OCR Developer (me)** | Image-to-text extraction for screenshot-based input |
| UI Developer | User interface design and implementation |
| Evaluation Developer | Model evaluation and metrics |

## Tools used

Python, Naive Bayes (scikit-learn), OCR library, TF-IDF vectorization

## Deliverables in this repo

- `project-report.pdf` — full team report
- `ocr-module/` — my OCR implementation
- `screenshots/` — system demo output

---
*Completed as part of SOF106: Principles of Artificial Intelligence, a 6-person team project. This repo documents my individual contribution (the OCR module) within the larger team deliverable.*
