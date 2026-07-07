// ============================================================
//  CREDIT SCORING SYSTEM — Frontend Logic
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initSlider();
    initForm();
    animateBarsOnReveal();
});

// ---- Tab Navigation ----
function initTabs() {
    const buttons = document.querySelectorAll('.tab-btn');
    const panels = document.querySelectorAll('.tab-panel');

    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;

            // Deactivate all
            buttons.forEach(b => b.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));

            // Activate selected
            btn.classList.add('active');
            const panel = document.getElementById(target);
            panel.classList.add('active');

            // Trigger bar animations on Analytics tab
            if (target === 'analytics') {
                setTimeout(() => animateBars(), 200);
            }
        });
    });
}

// ---- Slider Live Value ----
function initSlider() {
    const slider = document.getElementById('duration');
    const display = document.getElementById('duration-value');
    if (slider && display) {
        display.textContent = slider.value + ' months';
        slider.addEventListener('input', () => {
            display.textContent = slider.value + ' months';
        });
    }
}

// ---- Credit Application Form ----
function initForm() {
    const form = document.getElementById('credit-form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const btn = form.querySelector('.btn-primary');
        const resultContainer = document.getElementById('result-container');

        // Show loading state
        btn.classList.add('loading');
        btn.innerHTML = '<span class="spinner"></span> Analyzing...';
        resultContainer.innerHTML = '';

        const data = {
            full_name: document.getElementById('full_name').value,
            age: parseInt(document.getElementById('age').value),
            amount: parseInt(document.getElementById('amount').value),
            duration: parseInt(document.getElementById('duration').value),
            housing: document.getElementById('housing').value,
            purpose: document.getElementById('purpose').value
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            renderResult(result, resultContainer);
        } catch (error) {
            resultContainer.innerHTML = `
                <div class="result-card rejected">
                    <div class="result-title">⚠️ CONNECTION ERROR</div>
                    <div class="result-detail">Unable to reach the scoring engine. Please try again.</div>
                </div>
            `;
        } finally {
            btn.classList.remove('loading');
            btn.innerHTML = '🔍 Analyze Credit Application';
        }
    });
}

function renderResult(result, container) {
    const isApproved = result.decision === 'APPROVED';
    const cardClass = isApproved ? 'approved' : 'rejected';
    const icon = isApproved ? '🎉' : '❌';
    const title = isApproved ? 'APPLICATION APPROVED' : 'APPLICATION REJECTED';
    const note = isApproved
        ? 'Low Risk profile confirmed.'
        : 'Warning: Request exceeds safe portfolio baseline parameters.';

    container.innerHTML = `
        <div class="result-card ${cardClass}">
            <div class="result-title">${icon} ${title}</div>
            <div class="result-detail"><strong>Applicant:</strong> ${result.applicant_name}</div>
            <div class="result-detail"><strong>Requested Amount:</strong> $${result.amount.toLocaleString()}</div>
            <div class="result-detail"><strong>System Confidence Score:</strong> ${result.confidence}%</div>
            <div class="result-note">${note}</div>
        </div>
    `;

    // Scroll to result
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ---- Animate Performance & Feature Bars ----
function animateBars() {
    // Performance metric bars
    document.querySelectorAll('.perf-bar').forEach(bar => {
        const target = bar.dataset.value;
        bar.style.width = target + '%';
    });

    // Feature importance bars
    document.querySelectorAll('.feature-bar').forEach(bar => {
        const target = bar.dataset.value;
        bar.style.width = target + '%';
    });
}

function animateBarsOnReveal() {
    // If analytics tab is visible on load, animate
    const analyticsPanel = document.getElementById('analytics');
    if (analyticsPanel && analyticsPanel.classList.contains('active')) {
        setTimeout(() => animateBars(), 500);
    }
}
