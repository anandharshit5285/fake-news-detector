// Fake News Detector - Frontend JavaScript (Fixed Version)

// Session statistics
let sessionStats = {
    totalAnalyzed: 0,
    realNews: 0,
    fakeNews: 0,
    totalConfidence: 0
};

// DOM Elements
const newsInput = document.getElementById('newsInput');
const charCount = document.getElementById('charCount');
const wordCount = document.getElementById('wordCount');
const analyzeBtn = document.getElementById('analyzeBtn');
const detailedAnalysisBtn = document.getElementById('detailedAnalysisBtn');
const clearBtn = document.getElementById('clearBtn');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsContainer = document.getElementById('resultsContainer');
const errorMessage = document.getElementById('errorMessage');

// Sample news articles
const sampleNews = {
    real: `Scientists at MIT have developed a groundbreaking new solar panel technology that could revolutionize renewable energy. The new panels, which use a novel arrangement of photovoltaic cells, achieve 47% efficiency - nearly double that of conventional solar panels. Dr. Sarah Chen, lead researcher on the project, explained that the breakthrough came from studying how plants optimize photosynthesis. The team spent five years developing the technology, which they say could be commercially available within three years. Independent experts have verified the results, calling it a significant advancement in clean energy technology.`,
    
    fake: `BREAKING: Aliens have officially made contact with world governments! According to anonymous sources, extraterrestrial beings landed at Area 51 last night and held secret meetings with top officials. The aliens, reportedly from a distant galaxy, came to warn Earth about an impending disaster. Military personnel witnessed the entire event but have been ordered to remain silent. This information has been leaked by a brave whistleblower who wishes to remain anonymous. Share this before it gets deleted! The mainstream media won't report this because they're controlled by the government!`
};

// Update character and word count
if (newsInput) {
    newsInput.addEventListener('input', () => {
        const text = newsInput.value;
        if (charCount) charCount.textContent = text.length;
        if (wordCount) wordCount.textContent = text.trim().split(/\s+/).filter(word => word.length > 0).length;
    });
}

// Sample news buttons
document.querySelectorAll('.sample-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const type = btn.getAttribute('data-type');
        if (newsInput && sampleNews[type]) {
            newsInput.value = sampleNews[type];
            newsInput.dispatchEvent(new Event('input'));
            newsInput.focus();
            newsInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    });
});

// Clear button
if (clearBtn) {
    clearBtn.addEventListener('click', () => {
        if (newsInput) newsInput.value = '';
        if (charCount) charCount.textContent = '0';
        if (wordCount) wordCount.textContent = '0';
        hideResults();
        hideError();
    });
}

// Analyze button
if (analyzeBtn) {
    analyzeBtn.addEventListener('click', () => {
        analyzeNews(false);
    });
}

// Detailed analysis button
if (detailedAnalysisBtn) {
    detailedAnalysisBtn.addEventListener('click', () => {
        analyzeNews(true);
    });
}

// Main analysis function
async function analyzeNews(detailed = false) {
    if (!newsInput) return;
    
    const text = newsInput.value.trim();
    
    // Validation
    if (text.length < 10) {
        showError('Please enter at least 10 characters of news text.');
        return;
    }
    
    // Hide previous results
    hideResults();
    hideError();
    
    // Show loading
    showLoading();
    
    // Disable buttons
    setButtonsState(true);
    
    try {
        const endpoint = detailed ? '/analyze' : '/predict';
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }
        
        // Update session stats
        updateSessionStats(data);
        
        // Hide loading
        hideLoading();
        
        // Show results
        if (detailed) {
            displayDetailedResults(data);
        } else {
            displayResults(data);
        }
        
    } catch (error) {
        hideLoading();
        console.error('Analysis error:', error);
        showError('Error: ' + error.message + '. Make sure Flask server is running (python app.py)');
    } finally {
        setButtonsState(false);
    }
}

// Display quick results
function displayResults(data) {
    const isFake = data.label === 1;
    
    // Update badge
    const badge = document.getElementById('resultBadge');
    if (badge) {
        badge.textContent = data.prediction;
        badge.className = `badge ${isFake ? 'fake' : 'real'}`;
    }
    
    // Update confidence
    const confidenceBar = document.getElementById('confidenceBar');
    const confidencePercent = document.getElementById('confidencePercent');
    
    if (confidenceBar && confidencePercent) {
        confidenceBar.style.width = '0%';
        confidenceBar.className = `progress-fill ${isFake ? 'fake' : 'real'}`;
        
        setTimeout(() => {
            confidenceBar.style.width = `${data.confidence}%`;
            animateNumber(confidencePercent, 0, data.confidence, 1000);
        }, 100);
    }
    
    // Update stats
    const textLength = document.getElementById('textLength');
    const wordCountResult = document.getElementById('wordCountResult');
    if (textLength) textLength.textContent = data.text_length || 0;
    if (wordCountResult) wordCountResult.textContent = data.word_count || 0;
    
    // Show recommendation
    showRecommendation(data);
    
    // Hide detailed analysis
    const detailedResult = document.getElementById('detailedResult');
    if (detailedResult) detailedResult.style.display = 'none';
    
    // Show results container
    if (resultsContainer) {
        resultsContainer.style.display = 'block';
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Display detailed results
function displayDetailedResults(data) {
    // First show quick results
    displayResults(data);
    
    // Then show detailed analysis
    const detailedResult = document.getElementById('detailedResult');
    
    if (!detailedResult) return;
    
    // Update statistics
    if (data.statistics) {
        const origLength = document.getElementById('origLength');
        const cleanLength = document.getElementById('cleanLength');
        const totalWords = document.getElementById('totalWords');
        const uniqueWords = document.getElementById('uniqueWords');
        
        if (origLength) origLength.textContent = data.statistics.original_length;
        if (cleanLength) cleanLength.textContent = data.statistics.cleaned_length;
        if (totalWords) totalWords.textContent = data.statistics.word_count;
        if (uniqueWords) uniqueWords.textContent = data.statistics.unique_words;
    }
    
    // Update top keywords
    const keywordsContainer = document.getElementById('topKeywords');
    if (keywordsContainer && data.top_features && data.top_features.length > 0) {
        keywordsContainer.innerHTML = data.top_features.map(feature => `
            <div class="keyword-tag">
                <span>${feature.word}</span>
                <span class="keyword-score">${feature.score}</span>
            </div>
        `).join('');
    } else if (keywordsContainer) {
        keywordsContainer.innerHTML = '<p style="color: var(--text-secondary);">No significant keywords found</p>';
    }
    
    // Update cleaned text
    const cleanedTextContainer = document.getElementById('cleanedText');
    if (cleanedTextContainer && data.cleaned_text) {
        cleanedTextContainer.textContent = data.cleaned_text;
    } else if (cleanedTextContainer) {
        cleanedTextContainer.textContent = 'Processed text not available';
    }
    
    // Show detailed section
    detailedResult.style.display = 'block';
}

// Show recommendation based on result
function showRecommendation(data) {
    const recommendationText = document.getElementById('recommendationText');
    if (!recommendationText) return;
    
    const isFake = data.label === 1;
    const confidence = data.confidence;
    
    let message = '';
    
    if (isFake) {
        if (confidence >= 80) {
            message = 'This news article shows strong indicators of being fake or misleading. We recommend verifying this information from multiple trusted sources before sharing.';
        } else if (confidence >= 60) {
            message = 'This article shows some characteristics of fake news. Please cross-check this information with reputable news sources.';
        } else {
            message = 'The analysis suggests this might be fake news, but the confidence is moderate. Further verification from credible sources is recommended.';
        }
    } else {
        if (confidence >= 80) {
            message = 'This news article appears to be authentic based on our analysis. However, always verify important information from multiple trusted sources.';
        } else if (confidence >= 60) {
            message = 'This article shows characteristics of real news, though some caution is advised. Consider checking additional sources for complete accuracy.';
        } else {
            message = 'Our analysis suggests this is likely real news, but with moderate confidence. Verification from official sources is recommended.';
        }
    }
    
    recommendationText.textContent = message;
}

// Update session statistics
function updateSessionStats(data) {
    sessionStats.totalAnalyzed++;
    sessionStats.totalConfidence += data.confidence;
    
    if (data.label === 1) {
        sessionStats.fakeNews++;
    } else {
        sessionStats.realNews++;
    }
    
    // Update modal content if it exists
    updateStatsModal();
}

// Update statistics modal
function updateStatsModal() {
    const totalAnalyzed = document.getElementById('totalAnalyzed');
    const realNews = document.getElementById('realNews');
    const fakeNews = document.getElementById('fakeNews');
    const avgConfidence = document.getElementById('avgConfidence');
    
    if (totalAnalyzed) totalAnalyzed.textContent = sessionStats.totalAnalyzed;
    if (realNews) realNews.textContent = sessionStats.realNews;
    if (fakeNews) fakeNews.textContent = sessionStats.fakeNews;
    
    if (avgConfidence) {
        const avg = sessionStats.totalAnalyzed > 0 
            ? (sessionStats.totalConfidence / sessionStats.totalAnalyzed).toFixed(1)
            : 0;
        avgConfidence.textContent = avg + '%';
    }
}

// Animate number
function animateNumber(element, start, end, duration) {
    if (!element) return;
    
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current);
    }, 16);
}

// Helper functions
function showLoading() {
    if (loadingSpinner) loadingSpinner.style.display = 'block';
}

function hideLoading() {
    if (loadingSpinner) loadingSpinner.style.display = 'none';
}

function hideResults() {
    if (resultsContainer) resultsContainer.style.display = 'none';
}

function showError(message) {
    if (errorMessage) {
        errorMessage.style.display = 'flex';
        const errorText = document.getElementById('errorText');
        if (errorText) errorText.textContent = message;
    }
}

function hideError() {
    if (errorMessage) errorMessage.style.display = 'none';
}

function setButtonsState(disabled) {
    if (analyzeBtn) analyzeBtn.disabled = disabled;
    if (detailedAnalysisBtn) detailedAnalysisBtn.disabled = disabled;
    if (clearBtn) clearBtn.disabled = disabled;
}

// Modal functionality
const modal = document.getElementById('statsModal');
const statsLink = document.getElementById('statsLink');
const closeModal = document.querySelector('.close');

if (statsLink) {
    statsLink.addEventListener('click', (e) => {
        e.preventDefault();
        updateStatsModal();
        if (modal) modal.style.display = 'block';
    });
}

if (closeModal) {
    closeModal.addEventListener('click', () => {
        if (modal) modal.style.display = 'none';
    });
}

window.addEventListener('click', (e) => {
    if (e.target === modal) {
        if (modal) modal.style.display = 'none';
    }
});

// Smooth scrolling for navigation
document.querySelectorAll('.nav-links a').forEach(link => {
    link.addEventListener('click', (e) => {
        const href = link.getAttribute('href');
        if (href && href.startsWith('#') && href !== '#stats') {
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
                
                // Update active link
                document.querySelectorAll('.nav-links a').forEach(l => l.classList.remove('active'));
                link.classList.add('active');
            }
        }
    });
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to analyze
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (newsInput && newsInput.value.trim().length >= 10) {
            analyzeNews(false);
        }
    }
    
    // Ctrl/Cmd + Shift + Enter for detailed analysis
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'Enter') {
        if (newsInput && newsInput.value.trim().length >= 10) {
            analyzeNews(true);
        }
    }
    
    // Escape to close modal
    if (e.key === 'Escape' && modal && modal.style.display === 'block') {
        modal.style.display = 'none';
    }
});

// Intersection Observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe sections for fade-in effect
document.querySelectorAll('.process-card, .feature-item, .stat-item').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
});

// Welcome message in console
console.log('%c🚀 Fake News Detector Initialized!', 'color: #6C5CE7; font-size: 16px; font-weight: bold;');
console.log('%cKeyboard Shortcuts:', 'color: #00D2D3; font-size: 14px; font-weight: bold;');
console.log('%c  Ctrl/Cmd + Enter: Quick Analysis', 'color: #B8B8D1; font-size: 12px;');
console.log('%c  Ctrl/Cmd + Shift + Enter: Detailed Analysis', 'color: #B8B8D1; font-size: 12px;');
console.log('%c  Escape: Close Modal', 'color: #B8B8D1; font-size: 12px;');

// Server connection test on page load
window.addEventListener('load', () => {
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            console.log('%c✓ Server Connected Successfully!', 'color: #00B894; font-size: 14px; font-weight: bold;');
            console.log('%cModel Status:', 'color: #6C5CE7; font-size: 12px;', data);
        })
        .catch(error => {
            console.error('%c✗ Server Connection Failed!', 'color: #FF6B6B; font-size: 14px; font-weight: bold;');
            console.error('%cMake sure Flask server is running: python app.py', 'color: #FDCB6E; font-size: 12px;');
        });
});