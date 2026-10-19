// --- Initialization and Configuration ---
document.addEventListener('DOMContentLoaded', () => {
    feather.replace();

    // --- Example Setup ---
    const examples = [
        // 0: Initial point that challenges Newton, but still in a feasible region.
        { D0: 0.45, L0: 0.55, method: 'DFP', tol: 1e-6, max_iter: 200 },
        // 1: Distant corner, tests robustness and penalty recovery.
        { D0: 0.22, L0: 1.82, method: 'Newton', tol: 1e-6, max_iter: 200 },
        // 2: "Robust Start" version with few iterations for a quick visualization.
        { D0: 0.45, L0: 0.55, method: 'DFP', tol: 1e-6, max_iter: 20 },
        // 3: Point in a "narrow valley," modified to be numerically stable.
        { D0: 0.98, L0: 0.95, method: 'SD', tol: 1e-6, max_iter: 200 },
        // 4: Minimum volume violation, modified to be less severe and more stable.
        { D0: 0.8, L0: 1.2, method: 'Newton', tol: 1e-6, max_iter: 200 },
        // 5: Violates the maximum volume constraint.
        { D0: 1.1, L0: 1.5, method: 'DFP', tol: 1e-6, max_iter: 200 },
        // 6: Point in a corner (L=max), forces a large recovery.
        { D0: 0.32, L0: 1.98, method: 'SD', tol: 1e-6, max_iter: 200 },
        // 7: Point in another corner (D=max), also violating Vmin.
        { D0: 1.1, L0: 0.6, method: 'Newton', tol: 1e-6, max_iter: 200 },
    ];

    // --- DOM Elements ---
    const form = document.getElementById('opt-form');
    const runBtn = document.getElementById('run-btn');
    const runBtnText = document.getElementById('run-btn-text');
    
    const d0Input = document.getElementById('D0');
    const l0Input = document.getElementById('L0');
    const methodInput = document.getElementById('method');
    const tolInput = document.getElementById('tol');
    const maxIterInput = document.getElementById('max_iter');
    
    const tabs = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const methodSelectionWrapper = document.getElementById('method-selection-wrapper');

    const welcomeMessage = document.getElementById('welcome-message');
    
    const individualResultsCard = document.getElementById('individual-results-card');
    const individualContourCard = document.getElementById('individual-contour-card');
    const individualConvergenceCard = document.getElementById('individual-convergence-card');

    const comparisonContourCard = document.getElementById('comparison-contour-card');
    const comparisonConvergenceCard = document.getElementById('comparison-convergence-card');
    const comparisonTableCard = document.getElementById('comparison-table-card');
    
    const replayContourBtn = document.getElementById('replay-contour-btn');
    const replayConvergenceBtn = document.getElementById('replay-convergence-btn');

    const exampleBtns = document.querySelectorAll('.example-card');

    // Latest /compare payload, used by the replay buttons.
    let lastComparisonData = null;

    // --- Plotting Constants ---
    const feasibleRegionColor = 'rgba(46, 204, 113, 0.15)';
    const contourColorScale = 'Jet';
    const plotBgColor = '#1f2937';
    const plotFontColor = '#cbd5e1';
    const pathColors = {
        'SD': '#f1c40f', 'Newton': '#2ecc71', 'DFP': '#3498db'
    };
    
    const np = { array: (data) => {
        const arr = [...data];
        arr.getColumn = (col) => arr.map(row => row[col]);
        return arr;
    }};

    // --- Tab Logic ---
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`${tab.dataset.tab}-tab`).classList.add('active');
            if (methodSelectionWrapper) {
                methodSelectionWrapper.style.display = (tab.dataset.tab === 'comparison') ? 'none' : 'block';
            }
        });
    });

    // Ensures the method selector is in the correct state on page load
    const initialActiveTab = document.querySelector('.tab-btn.active');
    if (initialActiveTab && methodSelectionWrapper) {
        methodSelectionWrapper.style.display = (initialActiveTab.dataset.tab === 'comparison') ? 'none' : 'block';
    }

    // --- Execution Logic ---
    const getActiveMode = () => document.querySelector('.tab-btn.active').dataset.tab;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const params = {
            D0: d0Input.value, L0: l0Input.value, tol: tolInput.value,
            max_iter: maxIterInput.value, method: methodInput.value,
        };
        
        setUILoading(true);
        if(welcomeMessage) welcomeMessage.classList.add('hidden');

        const mode = getActiveMode();
        if (mode === 'individual') await runIndividualAnalysis(params);
        else await runComparisonAnalysis(params);
        
        setUILoading(false);
    });

    exampleBtns.forEach((btn, index) => {
        btn.addEventListener('click', () => {
            const exampleData = examples[index];
            d0Input.value = exampleData.D0;
            l0Input.value = exampleData.L0;
            methodInput.value = exampleData.method;
            tolInput.value = exampleData.tol;
            maxIterInput.value = exampleData.max_iter;
            form.requestSubmit();
        });
    });

    function setUILoading(isLoading) {
        if (!runBtn) return;
        runBtn.disabled = isLoading;
        if(runBtnText) runBtnText.textContent = isLoading ? 'Computing...' : 'Run';
        runBtn.classList.toggle('opacity-50', isLoading);
        runBtn.classList.toggle('cursor-not-allowed', isLoading);
    }

    // --- Analysis Functions ---
    async function runIndividualAnalysis(params) {
        [individualResultsCard, individualContourCard, individualConvergenceCard].forEach(c => c?.classList.add('hidden'));
        try {
            const response = await fetch('/optimize', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);

            populateIndividualResults(data);
            plotIndividualContour(data, params.method);
            plotIndividualConvergence(data);

            [individualResultsCard, individualContourCard, individualConvergenceCard].forEach(c => c?.classList.remove('hidden'));
        } catch (error) {
            console.error('Error in individual analysis:', error);
            alert(`Optimization failed: ${error.message}`);
        }
    }

    async function runComparisonAnalysis(params) {
        [comparisonContourCard, comparisonConvergenceCard, comparisonTableCard].forEach(c => c?.classList.add('hidden'));
        try {
            const response = await fetch('/compare', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);

            lastComparisonData = data; // Store the latest data

            populateComparisonTable(data.results);
            plotComparisonContour(data);
            plotComparisonConvergence(data.results);
            
            // Re-render Feather icons, especially for replay buttons
            feather.replace();

            [comparisonContourCard, comparisonConvergenceCard, comparisonTableCard].forEach(c => c?.classList.remove('hidden'));
        } catch (error) {
            console.error('Error in comparison analysis:', error);
            alert(`Comparison failed: ${error.message}`);
        }
    }
    
    // --- Data Population Functions ---
    function populateIndividualResults(data) {
        document.getElementById('res-d').textContent = data.final_D.toFixed(4);
        document.getElementById('res-l').textContent = data.final_L.toFixed(4);
        document.getElementById('res-cost').textContent = `$${data.final_cost.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        document.getElementById('res-iter').textContent = data.iterations;
        document.getElementById('res-eval').textContent = data.num_eval;
    }

    function populateComparisonTable(results) {
        const table = document.getElementById('comparison-results-table');
        const metrics = [
            { key: 'final_cost', name: 'Final Cost ($)', format: (v) => `$${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` },
            { key: 'iterations', name: 'Iterations' },
            { key: 'num_eval', name: 'Function Evaluations' },
            { key: 'num_grad', name: 'Gradient Evaluations' },
            { key: 'final_D', name: 'Final Diameter (D*) [m]', format: (v) => v.toFixed(4) },
            { key: 'final_L', name: 'Final Length (L*) [m]', format: (v) => v.toFixed(4) },
            { key: 'final_volume', name: 'Volume [m³]', format: (v) => v.toFixed(4) },
            { key: 'converged', name: 'Converged (‖∇f‖ ≤ ε)', format: (v) => (v ? 'yes' : 'no') },
        ];
        const methods = Object.keys(results).sort((a,b) => a.localeCompare(b));

        let headerHtml = '<thead><tr><th>Metric</th>';
        methods.forEach(m => headerHtml += `<th>${m}</th>`);
        headerHtml += '</tr></thead>';
        
        let bodyHtml = '<tbody>';
        metrics.forEach(metric => {
            bodyHtml += `<tr><td>${metric.name}</td>`;
            methods.forEach(method => {
                const value = results[method][metric.key];
                bodyHtml += `<td>${metric.format ? metric.format(value) : value}</td>`;
            });
            bodyHtml += '</tr>';
        });
        bodyHtml += '</tbody>';
        if (table) table.innerHTML = headerHtml + bodyHtml;
    }

    // --- Animation Functions ---
    // One pending frame per plot, so animating one plot never cancels the other.
    const animationFrameIds = {};

    function animateTraces(plotId, tracesToAnimate, traceIndices, animationSpeed = 20) {
        cancelAnimationFrame(animationFrameIds[plotId]);

        const plotDiv = document.getElementById(plotId);
        if (!plotDiv.data) return;

        // Clone traces to avoid modifying original data
        const animatedTraces = JSON.parse(JSON.stringify(tracesToAnimate));
        
        let maxDataLength = 0;
        animatedTraces.forEach(trace => {
            maxDataLength = Math.max(maxDataLength, trace.x.length);
            // Store the full data and clear for animation
            trace.full_x = trace.x;
            trace.full_y = trace.y;
            trace.x = [trace.full_x[0]];
            trace.y = [trace.full_y[0]];
        });

        // Determine the indices of the traces to be updated
        const indices = traceIndices || Array.from({ length: animatedTraces.length }, (_, i) => i + 2);

        // Update the data in the plot to the initial point
        for (let i = 0; i < animatedTraces.length; i++) {
            const traceIndex = indices[i];
            if(plotDiv.data[traceIndex]) {
                plotDiv.data[traceIndex].x = animatedTraces[i].x;
                plotDiv.data[traceIndex].y = animatedTraces[i].y;
            }
        }
        Plotly.redraw(plotId);

        let currentPoint = 1;

        function animationLoop() {
            if (currentPoint >= maxDataLength) {
                return;
            }

            for (let i = 0; i < animatedTraces.length; i++) {
                const trace = animatedTraces[i];
                if (currentPoint < trace.full_x.length) {
                    trace.x.push(trace.full_x[currentPoint]);
                    trace.y.push(trace.full_y[currentPoint]);
                }
            }
            
            const updateData = {
                x: animatedTraces.map(t => t.x),
                y: animatedTraces.map(t => t.y),
            };
            
            Plotly.restyle(plotId, updateData, indices);

            currentPoint++;
            animationFrameIds[plotId] = requestAnimationFrame(animationLoop);
        }

        setTimeout(() => {
            animationFrameIds[plotId] = requestAnimationFrame(animationLoop);
        }, animationSpeed);
    }

    replayContourBtn.addEventListener('click', () => {
        if (lastComparisonData) {
            plotComparisonContour(lastComparisonData);
        }
    });

    replayConvergenceBtn.addEventListener('click', () => {
        if (lastComparisonData) {
            plotComparisonConvergence(lastComparisonData.results);
        }
    });

    // --- Plotting Functions ---
    function getPlotLayout(title, xaxis_title, yaxis_title, xrange = null, yrange = null) {
        const layout = {
            title,
            xaxis: { title: xaxis_title, gridcolor: '#2b3a4e' },
            yaxis: { title: yaxis_title, gridcolor: '#2b3a4e' },
            plot_bgcolor: plotBgColor, paper_bgcolor: plotBgColor, font: { color: plotFontColor },
            legend: { x: 1, y: 1, xanchor: 'right', bgcolor: 'rgba(0,0,0,0.3)' },
        };
        if (xrange) {
            layout.xaxis.range = xrange;
        } else {
            layout.xaxis.autorange = true;
        }
        if (yrange) {
            layout.yaxis.range = yrange;
        } else {
            layout.yaxis.autorange = true;
        }
        return layout;
    }

    // Feasible set: V_min <= pi D^2 L / 4 <= V_max, D <= D_max, L <= L_max (limits come from the API).
    function getFeasibleRegionTrace(contourData) {
        const { v_min, v_max, d_max, l_max } = contourData.limits;
        const lAt = (V, D) => Math.min(4 * V / (Math.PI * D * D), l_max);
        const D_vals = contourData.d.filter(D => D <= d_max && lAt(v_min, D) < l_max);
        return {
            x: D_vals.concat(D_vals.slice().reverse()),
            y: D_vals.map(D => lAt(v_max, D)).concat(D_vals.slice().reverse().map(D => lAt(v_min, D))),
            fill: 'toself', fillcolor: feasibleRegionColor, line: { color: 'transparent' },
            type: 'scatter', hoverinfo: 'none', name: 'Feasible region'
        };
    }

    function plotIndividualContour(data, methodName) {
        const feasibleRegion = getFeasibleRegionTrace(data.contour);
        const contourTrace = { x: data.contour.d, y: data.contour.l, z: data.contour.cost, type: 'contour', colorscale: contourColorScale, showscale: false, contours: { coloring: 'lines' } };
        const history = np.array(data.history);
        const pathTrace = {
            x: history.getColumn(0), y: history.getColumn(1),
            mode: 'lines+markers', type: 'scatter', name: `${methodName} path`,
            marker: { color: pathColors[methodName] || '#FFF', size: 6, symbol: 'circle' },
            line: { color: pathColors[methodName] || '#FFF', width: 2 }
        };
        Plotly.newPlot('contour-plot', [feasibleRegion, contourTrace, pathTrace], getPlotLayout('Cost Contours and Trajectory', 'Diameter D [m]', 'Length L [m]'), {responsive: true});
    }
    
    function plotIndividualConvergence(data) {
        const trace = { x: Array.from({length: data.errors.length}, (_, i) => i), y: data.errors, mode: 'lines+markers', type: 'scatter' };
        const layout = getPlotLayout('Gradient Convergence', 'Iteration', '‖∇φ(D<sub>k</sub>, L<sub>k</sub>)‖ (current subproblem)');
        layout.yaxis.type = 'log';
        layout.yaxis.autorange = true;
        Plotly.newPlot('convergence-plot', [trace], layout, {responsive: true});
    }

    function plotComparisonContour(data) {
        const traces = [
            getFeasibleRegionTrace(data.contour), 
            { x: data.contour.d, y: data.contour.l, z: data.contour.cost, type: 'contour', colorscale: contourColorScale, showscale: false, contours: { coloring: 'lines' } }
        ];
        const methods = Object.keys(data.results).sort((a,b) => a.localeCompare(b));
        
        let all_x = [], all_y = [];
        const tracesToAnimate = [];

        // Add contour data for range calculation, ensuring the plot doesn't rescale
        if (data.contour && data.contour.d) {
            all_x.push(...data.contour.d);
        }
        if (data.contour && data.contour.l) {
            all_y.push(...data.contour.l);
        }

        methods.forEach(methodName => {
            const result = data.results[methodName];
            if (!result || !result.history || result.history.length === 0) return;

            const history = np.array(result.history);
            const x_hist = history.getColumn(0);
            const y_hist = history.getColumn(1);
            
            all_x.push(...x_hist);
            all_y.push(...y_hist);
            
            const trace = {
                x: [x_hist[0]], // Start with the first point
                y: [y_hist[0]],
                mode: 'lines+markers', type: 'scatter', name: `${methodName} path`,
                marker: { color: pathColors[methodName] || '#FFF', size: 6, symbol: 'circle' },
                line: { color: pathColors[methodName] || '#FFF', width: 2 }
            };
            traces.push(trace);
            
            tracesToAnimate.push({ x: x_hist, y: y_hist });
        });

        let layout;
        // Define the plot range based on all points (paths and contours) to avoid resizing
        if (all_x.length > 0 && all_y.length > 0) {
            const min_x = Math.min(...all_x);
            const max_x = Math.max(...all_x);
            const min_y = Math.min(...all_y);
        
            const padding_x = (max_x - min_x) * 0.1 || 0.1;
            // Add bottom padding, but keep the top fixed.
            const padding_y = (5 - min_y) * 0.1 || 0.1;
        
            const xrange = [min_x - padding_x, max_x + padding_x];
            const yrange = [min_y - padding_y, 5]; // Y-axis maximum fixed at 5
            
            layout = getPlotLayout('Trajectories', 'Diameter D [m]', 'Length L [m]', xrange, yrange);
        } else {
            layout = getPlotLayout('Trajectories', 'Diameter D [m]', 'Length L [m]');
        }
        
        Plotly.newPlot('comparison-contour-plot', traces, layout, {responsive: true}).then(() => {
            // The path traces start at index 2 (after region and contour)
            animateTraces('comparison-contour-plot', tracesToAnimate);
        });
    }

    function plotComparisonConvergence(results) {
        const traces = [];
        const tracesToAnimate = [];
        const methods = Object.keys(results).sort((a,b) => a.localeCompare(b));
        
        methods.forEach(methodName => {
            const result = results[methodName];
            if (result && result.errors && result.errors.length > 0) {
                const x_data = Array.from({length: result.errors.length}, (_, i) => i);
                const y_data = result.errors;

                traces.push({
                    x: [x_data[0]], // Start with the first point
                    y: [y_data[0]],
                    mode: 'lines+markers', type: 'scatter', name: methodName,
                    line: { color: pathColors[methodName] || '#FFFFFF' }
                });
                tracesToAnimate.push({ x: x_data, y: y_data });
            }
        });

        const layout = getPlotLayout('Gradient Convergence', 'Iteration', '‖∇φ(D<sub>k</sub>, L<sub>k</sub>)‖ (current subproblem)');
        layout.yaxis.type = 'log';
        layout.yaxis.autorange = true;
        
        // Add empty traces if there is no data to avoid errors
        if(traces.length === 0) {
            Plotly.newPlot('comparison-convergence-plot', [], layout, {responsive: true});
            return;
        }

        Plotly.newPlot('comparison-convergence-plot', traces, layout, {responsive: true}).then(() => {
            // Convergence traces have nothing before them, so indices are direct
            const traceIndices = Array.from({ length: tracesToAnimate.length }, (_, i) => i);
            animateTraces('comparison-convergence-plot', tracesToAnimate, traceIndices);
        });
    }
}); 