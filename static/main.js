// Sınıf renk ve emoji eşleştirmeleri
const CLASS_META = {
    'Iris-Setosa':     { emoji: '🌸', color: '#fb923c', cssClass: 'class-setosa',     dotColor: '#fb923c', glow: 'rgba(251,146,60,0.45)'  },
    'Iris-Versicolor': { emoji: '🌺', color: '#a78bfa', cssClass: 'class-versicolor', dotColor: '#a78bfa', glow: 'rgba(167,139,250,0.45)' },
    'Iris-Virginica':  { emoji: '🌼', color: '#34d399', cssClass: 'class-virginica',  dotColor: '#34d399', glow: 'rgba(52,211,153,0.45)'  },
};

function fillExample(sl, sw, pl, pw) {
    document.getElementById('sepal_length').value = sl;
    document.getElementById('sepal_width').value  = sw;
    document.getElementById('petal_length').value = pl;
    document.getElementById('petal_width').value  = pw;
}

document.addEventListener('DOMContentLoaded', () => {
    const form          = document.getElementById('prediction-form');
    const submitBtn     = document.getElementById('submit-btn');
    const btnText       = document.getElementById('btn-text');
    const loader        = document.getElementById('loader');
    const placeholder   = document.getElementById('result-placeholder');
    const resultContent = document.getElementById('result-content');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Loading state
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        submitBtn.disabled = true;

        const data = {
            sepal_length: parseFloat(document.getElementById('sepal_length').value),
            sepal_width:  parseFloat(document.getElementById('sepal_width').value),
            petal_length: parseFloat(document.getElementById('petal_length').value),
            petal_width:  parseFloat(document.getElementById('petal_width').value),
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!response.ok) throw new Error('Sunucu hatası');
            const result = await response.json();

            renderResult(result);

        } catch (err) {
            console.error(err);
            alert('Sunucu ile bağlantı kurulamadı. Uygulamanın çalıştığından emin olun.');
        } finally {
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
            submitBtn.disabled = false;
        }
    });

    function renderResult(result) {
        const className = result.class;
        const meta = CLASS_META[className] || { emoji: '🌿', color: '#64748b', cssClass: '', dotColor: '#64748b' };

        // Flower emoji & glow ring
        document.getElementById('flower-emoji').textContent = meta.emoji;
        document.getElementById('flower-card').style.boxShadow = `0 0 32px ${meta.color}22`;
        document.getElementById('glow-ring').style.background = meta.glow || meta.color;

        // Predicted class name
        const classEl = document.getElementById('predicted-class-name');
        classEl.textContent = className;
        classEl.className = `predicted-class ${meta.cssClass}`;

        // Confidence
        document.getElementById('confidence-value').textContent = `${result.confidence}%`;

        // Probability bars
        const barsContainer = document.getElementById('prob-bars');
        barsContainer.innerHTML = '';

        const allProbs = result.all_probs;
        Object.entries(allProbs).forEach(([name, pct]) => {
            const m = CLASS_META[name] || { dotColor: '#64748b' };

            const row = document.createElement('div');
            row.className = 'prob-row';
            row.innerHTML = `
                <div class="prob-dot" style="background:${m.dotColor}"></div>
                <span class="prob-name">${name}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="background:${m.dotColor};width:0%"></div>
                </div>
                <span class="prob-pct" style="color:${m.dotColor}">${pct}%</span>
            `;
            barsContainer.appendChild(row);

            // Animate bar
            requestAnimationFrame(() => {
                const fill = row.querySelector('.prob-bar-fill');
                fill.style.width = `${pct}%`;
            });
        });

        // Show result
        placeholder.classList.add('hidden');
        resultContent.classList.remove('hidden');
    }
});
