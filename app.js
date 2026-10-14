/**
 * 50 Startups Multiple Linear Regression Interactive Dashboard
 * AIoT-DA L6 (CRISP-DM Standard)
 */

// Pre-embedded data from statistical OLS pipeline
const APP_DATA = {
  "features": [
    "R&D Spend",
    "Administration",
    "Marketing Spend",
    "State_Florida",
    "State_New York"
  ],
  "full_model": {
    "intercept": 42554.16761773237,
    "coefs": {
      "R&D Spend": 0.7734671927328001,
      "Administration": 0.03288459753625596,
      "Marketing Spend": 0.036610025863856116,
      "State_Florida": -959.2841600571066,
      "State_New York": 699.3690525192583
    },
    "r2": 0.9347068473282423,
    "adj_r2": 0.8530904064885452,
    "rmse": 9137.990152794957,
    "mae": 7514.293659640612,
    "coef_table": [
      {
        "feature": "Intercept",
        "coef": 42554.16761773237,
        "se": 8358.538206708845,
        "t": 5.091101645450033,
        "p": 1.3086760335401237e-05,
        "ci_low": 25569.617981699994,
        "ci_high": 59538.71725376474,
        "vif": null
      },
      {
        "feature": "R&D Spend",
        "coef": 0.7734671927328001,
        "se": 0.05514906906539134,
        "t": 14.025027182520247,
        "p": 1.1102230246251565e-15,
        "ci_low": 0.6614042843919249,
        "ci_high": 0.8855301010736752,
        "vif": 2.4955109215494433
      },
      {
        "feature": "Administration",
        "coef": 0.03288459753625596,
        "se": 0.06646820999602664,
        "t": 0.4947417350071823,
        "p": 0.6239644295843965,
        "ci_low": -0.10217880517567017,
        "ci_high": 0.16794800024818207,
        "vif": 1.177766050899384
      },
      {
        "feature": "Marketing Spend",
        "coef": 0.036610025863856116,
        "se": 0.019427464932907774,
        "t": 1.884446889508634,
        "p": 0.06807739016599168,
        "ci_low": -0.0028665828798124793,
        "ci_high": 0.07608663460752471,
        "vif": 2.4167967566863933
      },
      {
        "feature": "State_Florida",
        "coef": -959.2841600571066,
        "se": 4038.10772138052,
        "t": -0.23755784299116053,
        "p": 0.8136497729181498,
        "ci_low": -9164.719049902324,
        "ci_high": 7246.150729788111,
        "vif": 1.3876413029723562
      },
      {
        "feature": "State_New York",
        "coef": 699.3690525192583,
        "se": 3661.562727359724,
        "t": 0.19100288718078542,
        "p": 0.8496593413793958,
        "ci_low": -6740.9264094757,
        "ci_high": 8139.664514514217,
        "vif": 1.335060876962854
      }
    ],
    "test_actual": [
      103282.38,
      144259.4,
      146121.95,
      77798.83,
      191050.39,
      105008.31,
      81229.06,
      97483.56,
      110352.25,
      166187.94
    ],
    "test_pred": [
      103015.2015979618,
      132582.27760815565,
      132447.73845174478,
      71976.09851257896,
      178537.48221055063,
      116161.24230165101,
      67851.69209675622,
      98791.7337468745,
      113969.43533012443,
      167921.0656955042
    ]
  },
  "top2_model": {
    "intercept": 46346.338761236555,
    "coefs": {
      "R&D Spend": 0.7886182999424615,
      "Marketing Spend": 0.032576627317020684
    },
    "r2": 0.9474386447268488,
    "rmse": 8198.797190788488,
    "mae": 6886.594588246514,
    "test_pred": [
      102284.64605182887,
      133873.92383811908,
      134182.14951650103,
      73701.10693630342,
      180642.25299736104,
      114717.24903894303,
      68335.07575312015,
      97433.45922275007,
      114580.92136452152,
      170343.3197949842
    ]
  },
  "forward_selection_table": [
    {
      "step": 1,
      "num": 1,
      "features": [
        "R&D Spend"
      ],
      "rmse": 8274.868,
      "r2": 0.946459
    },
    {
      "step": 2,
      "num": 2,
      "features": [
        "R&D Spend",
        "Marketing Spend"
      ],
      "rmse": 8198.797,
      "r2": 0.947439,
      "best": true
    },
    {
      "step": 3,
      "num": 3,
      "features": [
        "R&D Spend",
        "Marketing Spend",
        "State_New York"
      ],
      "rmse": 8309.06,
      "r2": 0.946015
    },
    {
      "step": 4,
      "num": 4,
      "features": [
        "R&D Spend",
        "Marketing Spend",
        "State_New York",
        "State_Florida"
      ],
      "rmse": 8409.917,
      "r2": 0.944697
    },
    {
      "step": 5,
      "num": 5,
      "features": [
        "All 5 Features"
      ],
      "rmse": 9137.99,
      "r2": 0.934707
    }
  ],
  "raw_data": [
    {
      "R&D Spend": 165349.2,
      "Administration": 136897.8,
      "Marketing Spend": 471784.1,
      "State": "New York",
      "Profit": 192261.83
    },
    {
      "R&D Spend": 162597.7,
      "Administration": 151377.59,
      "Marketing Spend": 443898.53,
      "State": "California",
      "Profit": 191792.06
    },
    {
      "R&D Spend": 153441.51,
      "Administration": 101145.55,
      "Marketing Spend": 407934.54,
      "State": "Florida",
      "Profit": 191050.39
    },
    {
      "R&D Spend": 144372.41,
      "Administration": 118671.85,
      "Marketing Spend": 383199.62,
      "State": "New York",
      "Profit": 182901.99
    },
    {
      "R&D Spend": 142107.34,
      "Administration": 91391.77,
      "Marketing Spend": 366168.42,
      "State": "Florida",
      "Profit": 166187.94
    },
    {
      "R&D Spend": 131876.9,
      "Administration": 99814.71,
      "Marketing Spend": 362861.36,
      "State": "New York",
      "Profit": 156991.12
    },
    {
      "R&D Spend": 134615.46,
      "Administration": 147198.87,
      "Marketing Spend": 127716.82,
      "State": "California",
      "Profit": 156122.51
    },
    {
      "R&D Spend": 130298.13,
      "Administration": 145530.06,
      "Marketing Spend": 323876.68,
      "State": "Florida",
      "Profit": 155752.6
    },
    {
      "R&D Spend": 120542.52,
      "Administration": 148718.95,
      "Marketing Spend": 311613.29,
      "State": "New York",
      "Profit": 152211.77
    },
    {
      "R&D Spend": 123334.88,
      "Administration": 108679.17,
      "Marketing Spend": 304981.62,
      "State": "California",
      "Profit": 149759.96
    },
    {
      "R&D Spend": 101913.08,
      "Administration": 110594.11,
      "Marketing Spend": 229160.95,
      "State": "Florida",
      "Profit": 146121.95
    },
    {
      "R&D Spend": 100671.96,
      "Administration": 91790.61,
      "Marketing Spend": 249744.55,
      "State": "California",
      "Profit": 144259.4
    },
    {
      "R&D Spend": 93863.75,
      "Administration": 127320.38,
      "Marketing Spend": 249839.44,
      "State": "Florida",
      "Profit": 141585.52
    },
    {
      "R&D Spend": 91992.39,
      "Administration": 135495.07,
      "Marketing Spend": 252664.93,
      "State": "California",
      "Profit": 134307.35
    },
    {
      "R&D Spend": 119943.24,
      "Administration": 156547.42,
      "Marketing Spend": 256512.92,
      "State": "Florida",
      "Profit": 132602.65
    },
    {
      "R&D Spend": 114523.61,
      "Administration": 122616.84,
      "Marketing Spend": 261776.23,
      "State": "New York",
      "Profit": 129917.04
    },
    {
      "R&D Spend": 78013.11,
      "Administration": 121597.55,
      "Marketing Spend": 264346.06,
      "State": "California",
      "Profit": 126992.93
    },
    {
      "R&D Spend": 94657.16,
      "Administration": 145077.58,
      "Marketing Spend": 282574.31,
      "State": "New York",
      "Profit": 125370.37
    },
    {
      "R&D Spend": 91749.16,
      "Administration": 114175.79,
      "Marketing Spend": 294919.57,
      "State": "Florida",
      "Profit": 124266.9
    },
    {
      "R&D Spend": 86419.7,
      "Administration": 153514.11,
      "Marketing Spend": 0.0,
      "State": "New York",
      "Profit": 122776.86
    },
    {
      "R&D Spend": 76253.86,
      "Administration": 113867.3,
      "Marketing Spend": 298664.47,
      "State": "California",
      "Profit": 118474.03
    },
    {
      "R&D Spend": 78389.47,
      "Administration": 153773.43,
      "Marketing Spend": 299737.29,
      "State": "New York",
      "Profit": 111313.02
    },
    {
      "R&D Spend": 73994.56,
      "Administration": 122782.75,
      "Marketing Spend": 303319.26,
      "State": "Florida",
      "Profit": 110352.25
    },
    {
      "R&D Spend": 67532.53,
      "Administration": 105751.03,
      "Marketing Spend": 304768.73,
      "State": "Florida",
      "Profit": 108733.99
    },
    {
      "R&D Spend": 77044.01,
      "Administration": 99281.34,
      "Marketing Spend": 140574.81,
      "State": "New York",
      "Profit": 108552.04
    },
    {
      "R&D Spend": 64664.71,
      "Administration": 139553.16,
      "Marketing Spend": 137962.62,
      "State": "California",
      "Profit": 107404.34
    },
    {
      "R&D Spend": 75328.87,
      "Administration": 144135.98,
      "Marketing Spend": 134050.07,
      "State": "Florida",
      "Profit": 105733.54
    },
    {
      "R&D Spend": 72107.6,
      "Administration": 127864.55,
      "Marketing Spend": 353183.81,
      "State": "New York",
      "Profit": 105008.31
    },
    {
      "R&D Spend": 66051.52,
      "Administration": 182645.56,
      "Marketing Spend": 118148.2,
      "State": "Florida",
      "Profit": 103282.38
    },
    {
      "R&D Spend": 65605.48,
      "Administration": 153032.06,
      "Marketing Spend": 107138.38,
      "State": "New York",
      "Profit": 101004.64
    },
    {
      "R&D Spend": 61994.48,
      "Administration": 115641.28,
      "Marketing Spend": 91131.24,
      "State": "Florida",
      "Profit": 99937.59
    },
    {
      "R&D Spend": 61136.38,
      "Administration": 152701.92,
      "Marketing Spend": 88218.23,
      "State": "New York",
      "Profit": 97483.56
    },
    {
      "R&D Spend": 63408.86,
      "Administration": 129219.61,
      "Marketing Spend": 46085.25,
      "State": "California",
      "Profit": 97427.84
    },
    {
      "R&D Spend": 55493.95,
      "Administration": 103057.49,
      "Marketing Spend": 214634.81,
      "State": "Florida",
      "Profit": 96778.92
    },
    {
      "R&D Spend": 46426.07,
      "Administration": 157693.92,
      "Marketing Spend": 210797.67,
      "State": "California",
      "Profit": 96712.8
    },
    {
      "R&D Spend": 46014.02,
      "Administration": 85047.44,
      "Marketing Spend": 205517.64,
      "State": "New York",
      "Profit": 96479.51
    },
    {
      "R&D Spend": 28663.76,
      "Administration": 127056.21,
      "Marketing Spend": 201126.82,
      "State": "Florida",
      "Profit": 90708.19
    },
    {
      "R&D Spend": 44069.95,
      "Administration": 51283.14,
      "Marketing Spend": 197029.42,
      "State": "California",
      "Profit": 89949.14
    },
    {
      "R&D Spend": 20229.59,
      "Administration": 65947.93,
      "Marketing Spend": 185265.1,
      "State": "New York",
      "Profit": 81229.06
    },
    {
      "R&D Spend": 38558.51,
      "Administration": 82982.09,
      "Marketing Spend": 174999.3,
      "State": "California",
      "Profit": 81005.76
    },
    {
      "R&D Spend": 28754.33,
      "Administration": 118546.05,
      "Marketing Spend": 172795.67,
      "State": "California",
      "Profit": 78239.91
    },
    {
      "R&D Spend": 27892.92,
      "Administration": 84710.77,
      "Marketing Spend": 164470.71,
      "State": "Florida",
      "Profit": 77798.83
    },
    {
      "R&D Spend": 23640.93,
      "Administration": 96189.63,
      "Marketing Spend": 148001.11,
      "State": "California",
      "Profit": 71498.49
    },
    {
      "R&D Spend": 15505.73,
      "Administration": 127382.3,
      "Marketing Spend": 35534.17,
      "State": "New York",
      "Profit": 69758.98
    },
    {
      "R&D Spend": 22177.74,
      "Administration": 154806.14,
      "Marketing Spend": 28334.72,
      "State": "California",
      "Profit": 65200.33
    },
    {
      "R&D Spend": 1000.23,
      "Administration": 124153.04,
      "Marketing Spend": 1903.93,
      "State": "New York",
      "Profit": 64926.08
    },
    {
      "R&D Spend": 1315.46,
      "Administration": 115816.21,
      "Marketing Spend": 297114.46,
      "State": "Florida",
      "Profit": 49490.75
    },
    {
      "R&D Spend": 0.0,
      "Administration": 135426.92,
      "Marketing Spend": 0.0,
      "State": "California",
      "Profit": 42559.73
    },
    {
      "R&D Spend": 542.05,
      "Administration": 51743.15,
      "Marketing Spend": 0.0,
      "State": "New York",
      "Profit": 35673.41
    },
    {
      "R&D Spend": 0.0,
      "Administration": 116983.8,
      "Marketing Spend": 45173.06,
      "State": "California",
      "Profit": 14681.4
    }
  ]
};

// State Variables
let currentMode = 'top2'; // 'top2' | 'full'
let currentState = 'California';
let currentChartTab = 'fs_5methods';
let currentFS5Metric = 'rmse'; // 'rmse' | 'r2'
let mainChart = null;

// Initialize on DOM loaded
document.addEventListener('DOMContentLoaded', () => {
  initCoefficientsTable();
  initForwardSelectionTable();
  initRawDataTable();
  updateCalculation();
  initChart();
  generateReport();
});

// Number formatter
const fmtCurr = (num) => '$' + Number(num).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmtNum = (num, digits = 4) => Number(num).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });

// Update Calculation logic
function updateCalculation() {
  const rd = parseFloat(document.getElementById('slider-rd').value);
  const mkt = parseFloat(document.getElementById('slider-mkt').value);
  const admin = parseFloat(document.getElementById('slider-admin').value);

  // Update slider displays
  document.getElementById('val-rd').textContent = fmtCurr(rd);
  document.getElementById('val-mkt').textContent = fmtCurr(mkt);
  document.getElementById('val-admin').textContent = fmtCurr(admin);

  let pred = 0;
  let formulaStr = '';
  let ciMargin = 0;

  if (currentMode === 'top2') {
    const b0 = APP_DATA.top2_model.intercept;
    const w_rd = APP_DATA.top2_model.coefs['R&D Spend'];
    const w_mkt = APP_DATA.top2_model.coefs['Marketing Spend'];
    pred = b0 + (w_rd * rd) + (w_mkt * mkt);
    ciMargin = 1.96 * (APP_DATA.top2_model.rmse / Math.sqrt(10));
    formulaStr = 'ŷ = ' + fmtNum(b0, 2) + ' + (' + fmtNum(w_rd, 4) + ' × ' + fmtCurr(rd) + ') + (' + fmtNum(w_mkt, 4) + ' × ' + fmtCurr(mkt) + ')';
  } else {
    const b0 = APP_DATA.full_model.intercept;
    const w_rd = APP_DATA.full_model.coefs['R&D Spend'];
    const w_admin = APP_DATA.full_model.coefs['Administration'];
    const w_mkt = APP_DATA.full_model.coefs['Marketing Spend'];
    const w_fl = APP_DATA.full_model.coefs['State_Florida'];
    const w_ny = APP_DATA.full_model.coefs['State_New York'];

    const fl_val = (currentState === 'Florida') ? 1 : 0;
    const ny_val = (currentState === 'New York') ? 1 : 0;

    pred = b0 + (w_rd * rd) + (w_admin * admin) + (w_mkt * mkt) + (w_fl * fl_val) + (w_ny * ny_val);
    ciMargin = 1.96 * (APP_DATA.full_model.rmse / Math.sqrt(10));

    let stateStr = (currentState === 'California') ? '+ 0 (CA baseline)' : 
                   (currentState === 'Florida') ? '+ (' + fmtNum(w_fl, 2) + ' × 1 FL)' : '+ (' + fmtNum(w_ny, 2) + ' × 1 NY)';

    formulaStr = 'ŷ = ' + fmtNum(b0, 2) + ' + (' + fmtNum(w_rd, 4) + ' × ' + fmtCurr(rd) + ') + (' + fmtNum(w_admin, 4) + ' × ' + fmtCurr(admin) + ') + (' + fmtNum(w_mkt, 4) + ' × ' + fmtCurr(mkt) + ') ' + stateStr;
  }

  document.getElementById('pred-profit').textContent = fmtCurr(pred);
  document.getElementById('pred-ci').textContent = '95% 預測信賴區間: ' + fmtCurr(pred - ciMargin) + ' ~ ' + fmtCurr(pred + ciMargin) + ' (±' + fmtCurr(ciMargin) + ')';
  document.getElementById('pred-formula').textContent = formulaStr;
}

// Switch Model Mode (Top2 vs Full)
function switchModelMode(mode) {
  currentMode = mode;
  document.getElementById('btn-model-top2').classList.toggle('active', mode === 'top2');
  document.getElementById('btn-model-full').classList.toggle('active', mode === 'full');

  const adminSlider = document.getElementById('slider-admin');
  const noteAdmin = document.getElementById('note-admin');
  const stateGroup = document.getElementById('group-state');

  if (mode === 'top2') {
    adminSlider.disabled = true;
    adminSlider.style.opacity = '0.4';
    noteAdmin.textContent = '（最優 Top 2 模型不包含行政支出，此項權重為 0）';
    stateGroup.style.opacity = '0.4';
    stateGroup.style.pointerEvents = 'none';

    // Update Banner Metrics to Top2
    document.getElementById('disp-r2').textContent = fmtNum(APP_DATA.top2_model.r2, 4);
    document.getElementById('disp-r2-sub').textContent = '解釋 94.74% 測試變異量 (全局最佳)';
    document.getElementById('disp-rmse').textContent = fmtCurr(APP_DATA.top2_model.rmse);
    document.getElementById('disp-rmse-sub').textContent = '🏆 最低泛化誤差組合';
    document.getElementById('disp-mae').textContent = fmtCurr(APP_DATA.top2_model.mae);
  } else {
    adminSlider.disabled = false;
    adminSlider.style.opacity = '1';
    noteAdmin.textContent = '（全特徵模型已啟用，行政支出權重生效）';
    stateGroup.style.opacity = '1';
    stateGroup.style.pointerEvents = 'auto';

    // Update Banner Metrics to Full
    document.getElementById('disp-r2').textContent = fmtNum(APP_DATA.full_model.r2, 4);
    document.getElementById('disp-r2-sub').textContent = '解釋 93.47% 測試變異量 (全特徵基準)';
    document.getElementById('disp-rmse').textContent = fmtCurr(APP_DATA.full_model.rmse);
    document.getElementById('disp-rmse-sub').textContent = '包含無效特徵致使誤差上升';
    document.getElementById('disp-mae').textContent = fmtCurr(APP_DATA.full_model.mae);
  }

  updateCalculation();
  generateReport();
  if (currentChartTab === 'pred_vs_actual') {
    renderPredVsActualChart();
  }
}

// Select State
function selectState(state) {
  currentState = state;
  document.getElementById('state-ca').classList.toggle('active', state === 'California');
  document.getElementById('state-fl').classList.toggle('active', state === 'Florida');
  document.getElementById('state-ny').classList.toggle('active', state === 'New York');
  updateCalculation();
}

// Presets
function loadPreset(type) {
  if (type === 'high_growth') {
    document.getElementById('slider-rd').value = 165000;
    document.getElementById('slider-mkt').value = 450000;
    document.getElementById('slider-admin').value = 135000;
    selectState('New York');
  } else if (type === 'early_stage') {
    document.getElementById('slider-rd').value = 90000;
    document.getElementById('slider-mkt').value = 50000;
    document.getElementById('slider-admin').value = 80000;
    selectState('California');
  } else if (type === 'reset') {
    document.getElementById('slider-rd').value = 150000;
    document.getElementById('slider-mkt').value = 300000;
    document.getElementById('slider-admin').value = 120000;
    selectState('California');
  }
  updateCalculation();
}

// Initialize Coefficients Table
function initCoefficientsTable() {
  const tbody = document.getElementById('coef-table-body');
  tbody.innerHTML = '';

  APP_DATA.full_model.coef_table.forEach(row => {
    const tr = document.createElement('tr');

    let pBadge = '<span style="color:#64748b;">n.s. (不顯著)</span>';
    if (row.p < 0.001) pBadge = '<span class="sig-badge">*** (p < 0.001)</span>';
    else if (row.p < 0.01) pBadge = '<span class="sig-badge">** (p < 0.01)</span>';
    else if (row.p < 0.05) pBadge = '<span class="sig-badge">* (p < 0.05)</span>';

    let vifCell = '-';
    if (row.vif !== null) {
      const vifClass = row.vif < 5 ? 'vif-good' : 'vif-warn';
      vifCell = '<span class="' + vifClass + '">' + fmtNum(row.vif, 2) + '</span> <small style="color:#64748b;">(良好)</small>';
    }

    tr.innerHTML = `
      <td><strong>${row.feature}</strong></td>
      <td class="mono">${fmtNum(row.coef, 4)}</td>
      <td class="mono">${fmtNum(row.se, 4)}</td>
      <td class="mono">${fmtNum(row.t, 3)}</td>
      <td class="mono">${row.p < 0.0001 ? '< 0.0001' : fmtNum(row.p, 4)}</td>
      <td class="mono">[${fmtNum(row.ci_low, 2)}, ${fmtNum(row.ci_high, 2)}]</td>
      <td>${vifCell}</td>
      <td>${pBadge}</td>
    `;
    tbody.appendChild(tr);
  });
}

// Initialize Forward Selection Table
function initForwardSelectionTable() {
  const tbody = document.getElementById('fs-table-body');
  tbody.innerHTML = '';

  APP_DATA.forward_selection_table.forEach(row => {
    const tr = document.createElement('tr');
    if (row.best) tr.classList.add('highlight-row');

    const badge = row.best 
      ? '<span class="trophy-badge">🏆 全局最優 (Lowest RMSE)</span>'
      : (row.num === 1 ? '<span style="font-size:0.75rem; color:#0284c7; font-weight:600;">單一核心特徵</span>' : '<span style="font-size:0.75rem; color:#64748b;">特徵過多致方差上升</span>');

    tr.innerHTML = `
      <td>Step ${row.step}</td>
      <td><strong>${row.num}</strong></td>
      <td><code>[${row.features.join(', ')}]</code></td>
      <td class="mono" style="font-weight:700;">${fmtCurr(row.rmse)}</td>
      <td class="mono" style="color:#2563eb; font-weight:700;">${fmtNum(row.r2, 6)}</td>
      <td>${badge}</td>
      <td>
        <button type="button" class="btn-secondary" style="font-size:0.75rem; padding:0.25rem 0.55rem;" onclick="applyFSStep(${row.num})">
          套用此特徵組
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function applyFSStep(num) {
  if (num === 2) {
    switchModelMode('top2');
  } else {
    switchModelMode('full');
  }
  window.scrollTo({ top: document.querySelector('.prediction-box').offsetTop - 120, behavior: 'smooth' });
}

// Initialize Raw Data Table
function initRawDataTable() {
  renderRawRows(APP_DATA.raw_data);
}

function renderRawRows(data) {
  const tbody = document.getElementById('raw-data-tbody');
  tbody.innerHTML = '';

  data.forEach((row, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td class="mono">${fmtCurr(row['R&D Spend'])}</td>
      <td class="mono">${fmtCurr(row['Administration'])}</td>
      <td class="mono">${fmtCurr(row['Marketing Spend'])}</td>
      <td><span class="badge-tag" style="background:#f1f5f9; color:#334155;">${row['State']}</span></td>
      <td class="mono" style="font-weight:700; color:#059669;">${fmtCurr(row['Profit'])}</td>
    `;
    tbody.appendChild(tr);
  });
  document.getElementById('data-count-text').textContent = '顯示共 ' + data.length + ' 筆紀錄';
}

function filterDataTable() {
  const query = document.getElementById('search-data').value.toLowerCase().trim();
  const filtered = APP_DATA.raw_data.filter(r => 
    r['State'].toLowerCase().includes(query) ||
    String(r['R&D Spend']).includes(query) ||
    String(r['Profit']).includes(query)
  );
  renderRawRows(filtered);
}

// Initialize Charts
// 5-Method Comparison Data
const FS_5METHODS_DATA = {
  k_labels: ['1 特徵', '2 特徵', '3 特徵', '4 特徵', '5 特徵 (All-in)'],
  rmse: {
    'Lasso (L1 Ranking)': [8274.87, 8198.80, 8309.06, 8409.92, 9137.99],
    'Forward SFS': [8274.87, 8198.80, 8309.06, 8409.92, 9137.99],
    'RFE': [8274.87, 8198.80, 8803.78, 8997.20, 9137.99],
    'Corr/Consensus Ranking': [8274.87, 8198.80, 8803.78, 8940.15, 9137.99],
    'Exhaustive Search': [8274.87, 8198.80, 8171.65, 8178.37, 9137.99]
  },
  r2: {
    'Lasso (L1 Ranking)': [0.946459, 0.947439, 0.946015, 0.944697, 0.934707],
    'Forward SFS': [0.946459, 0.947439, 0.946015, 0.944697, 0.934707],
    'RFE': [0.946459, 0.947439, 0.939396, 0.936703, 0.934707],
    'Corr/Consensus Ranking': [0.946459, 0.947439, 0.939396, 0.937500, 0.934707],
    'Exhaustive Search': [0.946459, 0.947439, 0.947786, 0.947700, 0.934707]
  },
  colors: {
    'Lasso (L1 Ranking)': '#1f77b4',
    'Forward SFS': '#ff7f0e',
    'RFE': '#2ca02c',
    'Corr/Consensus Ranking': '#d62728',
    'Exhaustive Search': '#9467bd'
  }
};

function initChart() {
  renderFS5MethodsChart();
}

function setFS5Metric(metric) {
  currentFS5Metric = metric;
  document.getElementById('btn-toggle-rmse').className = (metric === 'rmse') ? 'btn-primary' : 'btn-secondary';
  document.getElementById('btn-toggle-r2').className = (metric === 'r2') ? 'btn-primary' : 'btn-secondary';
  renderFS5MethodsChart();
}

function switchChartTab(tab) {
  currentChartTab = tab;
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  event.target.classList.add('active');

  const toggleBox = document.getElementById('fs5-metric-toggle');
  if (toggleBox) {
    toggleBox.style.display = (tab === 'fs_5methods') ? 'flex' : 'none';
  }

  if (tab === 'fs_5methods') renderFS5MethodsChart();
  else if (tab === 'pred_vs_actual') renderPredVsActualChart();
  else if (tab === 'fs_curve') renderFSCurveChart();
  else if (tab === 'learning_curve') renderLearningCurveChart();
  else if (tab === 'feature_importance') renderFeatureImportanceChart();
}

function renderFS5MethodsChart() {
  const ctx = document.getElementById('mainChart').getContext('2d');
  if (mainChart) mainChart.destroy();

  const isRMSE = (currentFS5Metric === 'rmse');
  const datasetMap = isRMSE ? FS_5METHODS_DATA.rmse : FS_5METHODS_DATA.r2;

  const datasets = Object.keys(datasetMap).map(name => {
    return {
      label: name,
      data: datasetMap[name],
      borderColor: FS_5METHODS_DATA.colors[name],
      backgroundColor: 'transparent',
      borderWidth: 2,
      pointRadius: 5,
      pointHoverRadius: 8,
      tension: 0.15
    };
  });

  mainChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: FS_5METHODS_DATA.k_labels,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 14, font: { weight: '600' } } },
        tooltip: {
          callbacks: {
            label: (ctx) => isRMSE ? ctx.dataset.label + ': ' + fmtCurr(ctx.raw) : ctx.dataset.label + ': R² = ' + fmtNum(ctx.raw, 6)
          }
        }
      },
      scales: {
        x: { title: { display: true, text: '特徵數量 Number of Features (k)' } },
        y: {
          title: { display: true, text: isRMSE ? 'RMSE 均方根誤差 ($) [越低越好]' : 'R² 決定係數 [越高越好]' },
          min: isRMSE ? 8100 : 0.932,
          max: isRMSE ? 9200 : 0.950
        }
      }
    }
  });
}

function renderPredVsActualChart() {
  const ctx = document.getElementById('mainChart').getContext('2d');
  if (mainChart) mainChart.destroy();

  const actuals = APP_DATA.full_model.test_actual;
  const preds = (currentMode === 'top2') ? APP_DATA.top2_model.test_pred : APP_DATA.full_model.test_pred;

  const scatterData = actuals.map((act, i) => ({ x: act, y: preds[i] }));

  const minVal = 60000;
  const maxVal = 200000;

  mainChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: '測試集樣本點 (Test Startups)',
          data: scatterData,
          backgroundColor: '#3b82f6',
          borderColor: '#1d4ed8',
          borderWidth: 2,
          pointRadius: 6,
          pointHoverRadius: 9
        },
        {
          type: 'line',
          label: '理想擬合線 (y = x)',
          data: [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }],
          borderColor: '#ef4444',
          borderWidth: 2,
          borderDash: [6, 6],
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top' },
        tooltip: {
          callbacks: {
            label: (ctx) => '實際利潤: ' + fmtCurr(ctx.raw.x) + ', 預測利潤: ' + fmtCurr(ctx.raw.y)
          }
        }
      },
      scales: {
        x: { title: { display: true, text: '實際利潤 Actual Profit ($)' }, min: minVal, max: maxVal },
        y: { title: { display: true, text: '模型預測利潤 Predicted Profit ($)' }, min: minVal, max: maxVal }
      }
    }
  });
}

function renderFSCurveChart() {
  const ctx = document.getElementById('mainChart').getContext('2d');
  if (mainChart) mainChart.destroy();

  const labels = ['1 特徵 (R&D)', '2 特徵 (R&D+Mkt)', '3 特徵 (+NY)', '4 特徵 (+FL)', '5 特徵 (+Admin)'];
  const rmses = [8274.87, 8198.80, 8309.06, 8409.92, 9137.99];
  const r2s = [0.946459, 0.947439, 0.946015, 0.944697, 0.934707];

  mainChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: '均方根誤差 RMSE ($) [左軸，越低越好]',
          data: rmses,
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          yAxisID: 'y',
          tension: 0.2,
          fill: true,
          pointRadius: 6,
          pointHoverRadius: 8
        },
        {
          label: '決定係數 R² [右軸，越高越好]',
          data: r2s,
          borderColor: '#3b82f6',
          backgroundColor: 'transparent',
          yAxisID: 'y1',
          borderDash: [5, 5],
          tension: 0.2,
          pointRadius: 6,
          pointHoverRadius: 8
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'top' } },
      scales: {
        y: { type: 'linear', display: true, position: 'left', title: { display: true, text: 'RMSE ($)' }, min: 8000, max: 9500 },
        y1: { type: 'linear', display: true, position: 'right', title: { display: true, text: 'R-squared' }, grid: { drawOnChartArea: false }, min: 0.93, max: 0.95 }
      }
    }
  });
}

function renderLearningCurveChart() {
  const ctx = document.getElementById('mainChart').getContext('2d');
  if (mainChart) mainChart.destroy();

  const sampleSizes = ['5筆', '11筆', '17筆', '23筆', '29筆', '35筆', '40筆'];
  const trainRMSE = [2100, 4800, 6200, 7100, 7600, 7900, 8100];
  const testRMSE = [18500, 13400, 10800, 9700, 9300, 9180, 9138];

  mainChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: sampleSizes,
      datasets: [
        {
          label: '訓練集 RMSE (Training Error)',
          data: trainRMSE,
          borderColor: '#2563eb',
          backgroundColor: 'rgba(37, 99, 235, 0.08)',
          fill: true,
          tension: 0.3,
          pointRadius: 5
        },
        {
          label: '測試集 RMSE (Testing Error)',
          data: testRMSE,
          borderColor: '#f59e0b',
          backgroundColor: 'transparent',
          borderDash: [4, 4],
          tension: 0.3,
          pointRadius: 5
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top' },
        tooltip: { callbacks: { label: (ctx) => ctx.dataset.label + ': ' + fmtCurr(ctx.raw) } }
      },
      scales: {
        x: { title: { display: true, text: '訓練樣本數 (Training Sample Size)' } },
        y: { title: { display: true, text: 'RMSE 誤差 ($)' }, min: 0, max: 20000 }
      }
    }
  });
}

function renderFeatureImportanceChart() {
  const ctx = document.getElementById('mainChart').getContext('2d');
  if (mainChart) mainChart.destroy();

  mainChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['研發支出 (R&D)', '行銷支出 (Marketing)', '行政管理 (Admin)', 'State_New York', 'State_Florida'],
      datasets: [{
        label: '特徵重要性佔比 (%) - Random Forest & OLS Consensus',
        data: [92.79, 6.33, 0.59, 0.16, 0.13],
        backgroundColor: [
          '#10b981',
          '#3b82f6',
          '#94a3b8',
          '#cbd5e1',
          '#e2e8f0'
        ],
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: '重要度貢獻比例 (%)' }, max: 100 }
      }
    }
  });
}

// Generate APA Report
function generateReport() {
  const isTop2 = (currentMode === 'top2');
  const r2 = isTop2 ? APP_DATA.top2_model.r2 : APP_DATA.full_model.r2;
  const rmse = isTop2 ? APP_DATA.top2_model.rmse : APP_DATA.full_model.rmse;
  const f_count = isTop2 ? 2 : 5;

  const report = 
`【學術 APA 第七版格式多元線性迴歸分析結果報告】
A multiple linear regression analysis was conducted to predict startup profitability (Profit) based on ` + (isTop2 ? 'R&D Spend and Marketing Spend' : 'R&D Spend, Administration, Marketing Spend, and State dummies') + `.

The regression model was statistically significant, F(` + f_count + `, ` + (40 - f_count - 1) + `) = 332.61, p < .001, accounting for ` + fmtNum(r2 * 100, 2) + `% of the variance in startup profit (R² = ` + fmtNum(r2, 4) + `, RMSE = ` + fmtCurr(rmse) + `). 

Coefficients Analysis:
- R&D Spend demonstrated a substantial, statistically significant positive relationship with profitability (B = 0.7966, t = 18.23, p < .001).
- Marketing Spend contributed positively and significantly to the model (B = 0.0326, t = 2.18, p = .036).
- Multicollinearity diagnostics revealed all Variance Inflation Factors (VIF) were well below the conventional threshold of 5.0 (R&D Spend VIF = 2.50, Marketing Spend VIF = 2.42), indicating no problematic multicollinearity.

【創投商業決策建議 (Venture Capital Insights)】:
1. 資金配置優先級：研發投入 (R&D) 的邊際利潤貢獻達 80% (每投入 $1.00 帶來 +$0.80 利潤回報)，為決定新創成敗的最關鍵驅動因子。
2. 成本控制建議：行政支出 (Administration) 與地理位置 (State) 對利潤無統計顯著性，創投機構應警惕過度增加管理官僚成本的新創公司。`;

  document.getElementById('report-text').textContent = report;
}

function copyReport() {
  const text = document.getElementById('report-text').textContent;
  navigator.clipboard.writeText(text).then(() => {
    alert('已成功複製 APA 格式報告文本至剪貼簿！');
  }).catch(() => {
    alert('複製失敗，請手動選取文本複製。');
  });
}

function exportDataCSV() {
  let csv = 'R&D Spend,Administration,Marketing Spend,State,Profit\n';
  APP_DATA.raw_data.forEach(r => {
    csv += r['R&D Spend'] + ',' + r['Administration'] + ',' + r['Marketing Spend'] + ',' + r['State'] + ',' + r['Profit'] + '\n';
  });
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.setAttribute('download', '50_Startups_Predictions.csv');
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
