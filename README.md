# Course Recommender System 🚀

A machine learning-powered web application that recommends personalized Coursera courses based on user input. Using advanced NLP techniques and cosine similarity algorithms, this system analyzes course descriptions, skills, and metadata to suggest highly relevant courses from a dataset of 3,500+ courses.

**Live App**: [https://smart-course-recommender-system.streamlit.app/](https://smart-course-recommender-system.streamlit.app/)

## 📋 Table of Contents
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [File Descriptions](#-file-descriptions)
- [Key Algorithms](#-key-algorithms)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **Smart Course Search**: Search courses by name, keywords, or required skills
- **Content-Based Filtering**: Intelligent recommendations based on course content similarity
- **Multi-Attribute Analysis**: Considers course name, description, skills, difficulty level, and ratings
- **Text Preprocessing**: Advanced NLP including stemming and text normalization
- **Interactive Web Interface**: User-friendly Streamlit-based dashboard
- **Fast Recommendations**: Pre-computed similarity matrices for instant results
- **Course Metadata**: Display comprehensive course information including university, difficulty, ratings, and URLs

## 🧠 How It Works

### Algorithm Overview
1. **Data Preprocessing**: Clean and normalize course text data
2. **Feature Engineering**: Combine multiple course attributes into tag vectors
3. **Stemming**: Reduce words to their root forms using Porter Stemmer
4. **Vectorization**: Convert text to numerical vectors using CountVectorizer
5. **Similarity Calculation**: Compute cosine similarity between all course pairs
6. **Recommendation**: Return top N most similar courses to user selection

### Process Flow
```
User Input
    ↓
Search/Match Course
    ↓
Fetch Similarity Scores
    ↓
Rank Recommendations
    ↓
Display Results with Details
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.x |
| **ML/NLP** | scikit-learn, NLTK |
| **Data Processing** | pandas, NumPy |
| **Web Framework** | Streamlit |
| **Serialization** | pickle |
| **Data Format** | CSV |

### Dependencies
```
numpy - Numerical computing
pandas - Data manipulation
scikit-learn - Machine learning
nltk - Natural language processing
streamlit - Web interface
matplotlib - Data visualization
seaborn - Statistical visualization
requests - HTTP library
```

## 📁 Project Structure

```
Course-Recommendation-System/
├── Data/
│   └── Coursera.csv                          # Raw dataset (3,500+ courses)
├── models/
│   ├── courses.pkl                           # Processed course dataframe
│   ├── course_list.pkl                       # Course metadata dictionary
│   └── similarity.pkl                        # Pre-computed similarity matrix
├── CourseRecommendationSystem.py              # Data preprocessing & model training
├── main.py                                   # Streamlit web application
├── requirements.txt                          # Python dependencies
├── README.md                                 # Project documentation
└── myenv/                                    # Python virtual environment
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step-by-Step Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/abhijeetballabh/Course-Recommendation-System.git
   cd Course-Recommendation-System
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   # Windows
   python -m venv myenv
   .\myenv\Scripts\activate

   # Mac/Linux
   python3 -m venv myenv
   source myenv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**:
   ```bash
   python -c "import pandas, sklearn, streamlit; print('All dependencies installed!')"
   ```

## 📖 Usage

### Option 1: Train Model & Run App (Complete Workflow)

1. **Preprocess Data & Train Model**:
   ```bash
   python CourseRecommendationSystem.py
   ```
   This will:
   - Load and clean the Coursera dataset
   - Apply text preprocessing and stemming
   - Create similarity matrix
   - Save model files in `models/` directory

2. **Launch Web Application**:
   ```bash
   streamlit run main.py
   ```
   The app will open at `http://localhost:8501`

### Option 2: Quick Start (Using Pre-trained Models)

If model files already exist in `models/`, simply run:
```bash
streamlit run main.py
```

### Using the Web Application

1. **Search for a Course**: Enter a course name in the search bar
2. **View Recommendations**: Click "Get Recommendations" to find similar courses
3. **Explore Course Details**: 
   - Course name and university
   - Difficulty level
   - Course rating
   - Related skills
   - Direct link to course
4. **Filter Results**: Use the difficulty slider to filter recommendations

## 📄 File Descriptions

### `CourseRecommendationSystem.py`
**Purpose**: Data preprocessing and model training

**Key Functions**:
- `preprocess_data()`: Loads CSV, cleans text, applies stemming
- `create_similarity_matrix()`: Vectorizes text and computes cosine similarity
- `save_resources()`: Serializes model artifacts to pickle files
- `main()`: Orchestrates the pipeline

**Output**: Saves 3 pickle files to `models/` directory

### `main.py`
**Purpose**: Streamlit web application interface

**Key Features**:
- Interactive search interface
- Course recommendations display
- Course details and metadata
- Difficulty level filtering
- Responsive layout with columns

**Cache**: Uses Streamlit's caching for performance optimization

### `Data/Coursera.csv`
**Dataset Columns**:
- Course Name
- University
- Difficulty Level
- Course Rating
- Course URL
- Course Description
- Skills

## 🔬 Key Algorithms

### Porter Stemming
Reduces inflected words to their root form:
- "running" → "run"
- "courses" → "cours"
- "learning" → "learn"

### CountVectorizer
Converts text documents to token count matrix:
- Max features: 5,000 most frequent terms
- Stop words: English stop words removed
- Sparse matrix format for efficiency

### Cosine Similarity
Measures similarity between course vectors:
```
similarity = cos(θ) = (A · B) / (|A| × |B|)
Range: 0 (dissimilar) to 1 (identical)
```

## 🔮 Future Enhancements

- [ ] Collaborative filtering based on user ratings
- [ ] Deep learning models (BERT, transformers)
- [ ] User preference learning
- [ ] Course difficulty prediction
- [ ] Multi-language support
- [ ] Real-time data updates from Coursera API
- [ ] User ratings and feedback system
- [ ] Advanced filtering (price, duration, language)
- [ ] Visualization of course relationships
- [ ] Recommendation explanation

