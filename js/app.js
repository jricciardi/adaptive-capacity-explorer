/* ============================================================
   Adaptive Capacity Self-Assessment Tool
   Application logic — IIFE, no build step
   ============================================================ */
(function () {
  'use strict';

  /* --------------------------------------------------------
     CONFIG
     -------------------------------------------------------- */
  const CONFIG = {
    dataFiles: {
      benchmarks: 'data/occupation_benchmarks.json',
      filterTree: 'data/occupation_filter_tree.json',
      density:    'data/density_scores.json'
    },

    search: {
      debounceMs: 150,
      minChars: 2,
      maxResults: 15
    },

    // C3: 5 financial runway bands
    runway: {
      bands: [
        { max: 3,  cls: 'indicator-danger',         label: 'That\u2019s a tight window. In practical terms, building even a few more months changes how many things you can say no to.' },
        { max: 6,  cls: 'indicator-warning',         label: 'A real buffer, but not a long one. Enough to be somewhat selective, not enough to be patient.' },
        { max: 12, cls: 'indicator-success',         label: 'Solid ground. The research says this is where the math starts working in your favor. You can afford to search for fit rather than searching for survival.' },
        { max: 18, cls: 'indicator-success-strong',  label: 'You\u2019ve bought yourself real freedom. At this level, you can afford to wait for something genuinely right. The data says people who can wait, land better.' },
        { max: Infinity, cls: 'indicator-success-extra', label: 'That\u2019s serious runway. You\u2019re in a position where a transition is a strategic choice, not an emergency response.' }
      ]
    },

    // C4: Layperson-friendly age band language
    age: {
      bands: [
        { max: 25,  label: 'You\u2019re early. Everything you build now \u2014skills, savings, network\u2014 compounds for decades.' },
        { max: 35,  label: 'Prime building years. The skills and savings you stack now are the adaptive capacity you\u2019ll draw on later.' },
        { max: 45,  label: 'You\u2019ve got a track record and you\u2019ve got time. This is the window where deliberate skill-building has the highest payoff relative to effort.' },
        { max: 55,  label: 'Your experience is genuinely valuable, the question is how to deploy it. Adjacent moves that leverage what you know tend to land better than clean-sheet pivots from here.' },
        { max: Infinity, label: 'The research doesn\u2019t sugarcoat this: reemployment gets harder. But preparation changes the odds significantly, and the factors you can control (savings, skills, network) matter more now than at any other point in your career.' }
      ]
    },

    // Wealth Z-score: log-transform
    wealth: {
      medianMonths: 3,  // reference point
      scale: null        // computed: log(6+1) - log(3+1)
    },

    // Age Z-score: linear
    ageCalc: {
      center: 40,
      scale: 15
    },

    zCap: 2.0,   // winsorization cap for user-input Z-scores

    componentColors: {
      high: '#047857',   // green ≥ 60th
      mid:  '#d97706',   // amber ≥ 40th
      low:  '#dc2626'    // red < 40th
    },

    // C1: Descriptive tier thresholds
    scoreTiers: [
      { min: 75, label: 'Strong Position',   cls: 'tier-strong' },
      { min: 50, label: 'Solid Foundation',   cls: 'tier-above' },
      { min: 25, label: 'Room to Build',      cls: 'tier-moderate' },
      { min: 0,  label: 'Early Stages',       cls: 'tier-below' }
    ]
  };

  // Pre-compute wealth scale
  CONFIG.wealth.scale = Math.log(6 + 1) - Math.log(CONFIG.wealth.medianMonths + 1);

  /* --------------------------------------------------------
     STATE
     -------------------------------------------------------- */
  const state = {
    // Data
    benchmarks: null,
    filterTree: null,
    density: null,

    // Derived on load
    searchArray: [],      // [{soc, title, titleLower}]
    cbsaArray: [],        // [{code, name, nameLower, employment}]

    // Quiz state
    quiz: {
      q1: null,
      q2: null,
      q3: null,
      q4: null
    },

    // User inputs
    selectedOccupation: null,  // SOC code
    financialRunway: null,     // months
    selectedMetro: null,       // CBSA code
    selectedMetroName: null,
    age: null,

    // Current step
    currentStep: 1,

    // Results
    results: null,

    // Chart.js instances
    charts: {
      scatter: null,
      components: null
    }
  };

  /* --------------------------------------------------------
     DOM REFERENCES
     -------------------------------------------------------- */
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const DOM = {
    overlay:        $('#loading-overlay'),
    hero:           $('#hero'),
    explainer:      $('#explainer'),
    assessment:     $('#assessment'),
    results:        $('#results'),

    btnStart:       $('#btn-start'),
    btnLearn:       $('#btn-learn'),
    btnPrev:        $('#btn-prev'),
    btnNext:        $('#btn-next'),
    btnResults:     $('#btn-results'),
    btnRestart:     $('#btn-restart'),
    btnStartOver:   $('#btn-start-over'),

    progressFill:   $('#progress-fill'),
    progressLabel:  $('#progress-label'),

    // Tabs
    tabQuiz:        $('#tab-quiz'),
    tabSearch:      $('#tab-search'),
    panelQuiz:      $('#panel-quiz'),
    panelSearch:    $('#panel-search'),

    // Quiz
    quizQ1:         $('#quiz-q1'),
    quizQ2:         $('#quiz-q2'),
    quizQ3:         $('#quiz-q3'),
    quizQ4:         $('#quiz-q4'),
    quizResults:    $('#quiz-results'),

    // Search
    searchInput:    $('#search-input'),
    searchResults:  $('#search-results'),

    // Occupation context
    occContext:     $('#occupation-context'),

    // Steps
    stepOccupation: $('#step-occupation'),
    stepFinancial:  $('#step-financial'),
    stepLocation:   $('#step-location'),
    stepAge:        $('#step-age'),

    // Financial
    inputRunway:    $('#input-runway'),
    runwayIndicator: $('#runway-indicator'),

    // Location
    inputMetro:     $('#input-metro'),
    metroListbox:   $('#metro-listbox'),
    locationContext: $('#location-context'),

    // Age
    inputAge:       $('#input-age'),
    ageContext:      $('#age-context'),

    // Results
    resultsHeadline:  $('#results-headline'),
    resultsBreakdown: $('#results-breakdown'),
    resultsCallouts:  $('#results-callouts'),
    roadmapContent:   $('#roadmap-content'),
    neighborsContent: $('#neighbors-content'),

    // Charts
    chartScatter:    $('#chart-scatter'),
    chartComponents: $('#chart-components')
  };

  /* --------------------------------------------------------
     UTILITIES
     -------------------------------------------------------- */
  function clamp(val, lo, hi) { return Math.max(lo, Math.min(hi, val)); }

  function formatNumber(n) {
    if (n == null) return '—';
    return n.toLocaleString('en-US');
  }

  function formatPct(n, decimals) {
    if (n == null) return '—';
    decimals = decimals != null ? decimals : 0;
    return n.toFixed(decimals) + '%';
  }

  function formatWage(w, topCoded) {
    if (w == null) return '—';
    var str = '$' + formatNumber(w);
    if (topCoded) str += '+';
    return str;
  }

  function formatGrowth(r) {
    if (r == null) return '—';
    var sign = r >= 0 ? '+' : '';
    return sign + (r * 100).toFixed(1) + '%';
  }

  function ordinal(n) {
    n = Math.round(n);
    var s = ['th', 'st', 'nd', 'rd'];
    var v = n % 100;
    return n + (s[(v - 20) % 10] || s[v] || s[0]);
  }

  // Normal CDF approximation (Abramowitz & Stegun)
  function normalCDF(z) {
    if (z < -6) return 0;
    if (z > 6) return 1;
    var t = 1 / (1 + 0.2316419 * Math.abs(z));
    var d = 0.3989422804014327; // 1/sqrt(2*pi)
    var p = d * Math.exp(-z * z / 2);
    var poly = ((((1.330274429 * t - 1.821255978) * t + 1.781477937) * t - 0.356563782) * t + 0.319381530) * t;
    return z >= 0 ? 1 - p * poly : p * poly;
  }

  function debounce(fn, ms) {
    var timer;
    return function () {
      var ctx = this, args = arguments;
      clearTimeout(timer);
      timer = setTimeout(function () { fn.apply(ctx, args); }, ms);
    };
  }

  function escapeHtml(s) {
    var el = document.createElement('span');
    el.textContent = s;
    return el.innerHTML;
  }

  function getScoreTier(pct) {
    for (var i = 0; i < CONFIG.scoreTiers.length; i++) {
      if (pct >= CONFIG.scoreTiers[i].min) return CONFIG.scoreTiers[i];
    }
    return CONFIG.scoreTiers[CONFIG.scoreTiers.length - 1];
  }

  /* --------------------------------------------------------
     DATA LOADER
     -------------------------------------------------------- */
  const DataLoader = {
    async loadAll() {
      try {
        const [benchmarks, filterTree, density] = await Promise.all([
          fetch(CONFIG.dataFiles.benchmarks).then(r => { if (!r.ok) throw new Error('Failed to load benchmarks: ' + r.status); return r.json(); }),
          fetch(CONFIG.dataFiles.filterTree).then(r => { if (!r.ok) throw new Error('Failed to load filter tree: ' + r.status); return r.json(); }),
          fetch(CONFIG.dataFiles.density).then(r => { if (!r.ok) throw new Error('Failed to load density scores: ' + r.status); return r.json(); })
        ]);

        state.benchmarks = benchmarks;
        state.filterTree = filterTree;
        state.density = density;

        // Build search array from search_index
        var idx = filterTree.search_index;
        state.searchArray = Object.keys(idx).map(function (soc) {
          return { soc: soc, title: idx[soc].title, titleLower: idx[soc].title.toLowerCase() };
        });

        // Build CBSA array
        var lookup = density.cbsa_lookup;
        state.cbsaArray = Object.keys(lookup).map(function (code) {
          return {
            code: code,
            name: lookup[code].name,
            nameLower: lookup[code].name.toLowerCase(),
            employment: lookup[code].total_employment
          };
        });
        // Sort by employment descending (most likely selections first)
        state.cbsaArray.sort(function (a, b) { return b.employment - a.employment; });

      } catch (err) {
        throw err;
      }
    }
  };

  /* --------------------------------------------------------
     QUIZ ENGINE
     -------------------------------------------------------- */
  const QuizEngine = {
    init() {
      this.renderQ1();
    },

    renderQ1() {
      var q = state.filterTree.questions.q1;
      DOM.quizQ1.innerHTML = this._buildQuestion(q.text, q.options, 'q1');
      DOM.quizQ1.hidden = false;
      DOM.quizQ2.hidden = true;
      DOM.quizQ3.hidden = true;
      DOM.quizQ4.hidden = true;
      DOM.quizResults.hidden = true;
    },

    renderQ2() {
      var q1 = state.quiz.q1;
      var branch = state.filterTree.questions.q2.branches[q1];
      if (!branch) return;
      DOM.quizQ2.innerHTML = this._buildQuestion(branch.text, branch.options, 'q2');
      DOM.quizQ2.hidden = false;
      DOM.quizQ3.hidden = true;
      DOM.quizQ4.hidden = true;
      DOM.quizResults.hidden = true;
    },

    // Fix 2: Show Q3 even with 1 option, pre-selected with note
    renderQ3() {
      var path = state.quiz.q1 + '.' + state.quiz.q2;
      var validOpts = state.filterTree.valid_q3_options[path];
      if (!validOpts || validOpts.length === 0) {
        this._tryShowLeaf();
        return;
      }

      var allQ3 = state.filterTree.questions.q3;
      var filtered = allQ3.options.filter(function (o) { return validOpts.indexOf(o.id) !== -1; });

      if (filtered.length === 1) {
        // Show the single option pre-selected with a note, rather than auto-skipping
        state.quiz.q3 = filtered[0].id;
        var html = '<p class="quiz-prompt">' + escapeHtml(allQ3.text) + '</p>';
        html += '<p class="hint" style="margin-bottom: var(--space-sm);">Only one education level matches your selections:</p>';
        html += '<div class="quiz-options" role="radiogroup">';
        html += '<button class="quiz-option selected" data-question="q3" data-value="' + filtered[0].id + '" type="button" role="radio" aria-checked="true">';
        html += escapeHtml(filtered[0].label);
        html += '</button>';
        html += '</div>';
        DOM.quizQ3.innerHTML = html;
        DOM.quizQ3.hidden = false;
        DOM.quizQ4.hidden = true;
        DOM.quizResults.hidden = true;
        // Auto-advance after a brief pause so user sees the selection
        setTimeout(function () { QuizEngine.afterQ3(); }, 400);
        return;
      }

      DOM.quizQ3.innerHTML = this._buildQuestion(allQ3.text, filtered, 'q3');
      DOM.quizQ3.hidden = false;
      DOM.quizQ4.hidden = true;
      DOM.quizResults.hidden = true;
    },

    afterQ3() {
      var path = state.quiz.q1 + '.' + state.quiz.q2 + '.' + state.quiz.q3;
      // Check if Q4 branch exists
      var q4Branches = state.filterTree.questions.q4.branches;
      if (q4Branches[path]) {
        this.renderQ4(path);
      } else {
        // No Q4 — show leaf occupations directly
        this._showLeafOccupations(path);
      }
    },

    renderQ4(path) {
      var branch = state.filterTree.questions.q4.branches[path];
      DOM.quizQ4.innerHTML = this._buildQuestion(branch.text, branch.options, 'q4');
      DOM.quizQ4.hidden = false;
      DOM.quizResults.hidden = true;
    },

    _tryShowLeaf() {
      var path = state.quiz.q1 + '.' + state.quiz.q2;
      var leaves = state.filterTree.leaves;
      var matching = Object.keys(leaves).filter(function (k) { return k.indexOf(path) === 0; });
      if (matching.length === 1) {
        this._showLeafOccupations(matching[0]);
      }
    },

    _showLeafOccupations(leafKey) {
      var leaf = state.filterTree.leaves[leafKey];
      if (!leaf) return;

      var html = '<p class="quiz-prompt">Select the occupation that best matches your current role:</p>';
      html += '<div class="occupation-list">';
      leaf.occupations.forEach(function (occ) {
        html += '<button class="occupation-option" data-soc="' + occ.soc + '" type="button">';
        html += '<span class="occ-title">' + escapeHtml(occ.title) + '</span>';
        if (occ.description) {
          html += '<span class="occ-desc">' + escapeHtml(occ.description) + '</span>';
        }
        html += '</button>';
      });
      html += '</div>';

      DOM.quizResults.innerHTML = html;
      DOM.quizResults.hidden = false;

      // Bind occupation clicks
      DOM.quizResults.querySelectorAll('.occupation-option').forEach(function (btn) {
        btn.addEventListener('click', function () {
          DOM.quizResults.querySelectorAll('.occupation-option').forEach(function (b) { b.classList.remove('selected'); });
          btn.classList.add('selected');
          state.selectedOccupation = btn.dataset.soc;
          UI.showOccupationContext();
          UI.validateStep();
        });
      });
    },

    handleSelection(question, value) {
      state.quiz[question] = value;

      // Clear downstream
      if (question === 'q1') {
        state.quiz.q2 = null;
        state.quiz.q3 = null;
        state.quiz.q4 = null;
        state.selectedOccupation = null;
        DOM.occContext.hidden = true;
        this.renderQ2();
      } else if (question === 'q2') {
        state.quiz.q3 = null;
        state.quiz.q4 = null;
        state.selectedOccupation = null;
        DOM.occContext.hidden = true;
        this.renderQ3();
      } else if (question === 'q3') {
        state.quiz.q4 = null;
        state.selectedOccupation = null;
        DOM.occContext.hidden = true;
        this.afterQ3();
      } else if (question === 'q4') {
        state.selectedOccupation = null;
        DOM.occContext.hidden = true;
        var path = state.quiz.q1 + '.' + state.quiz.q2 + '.' + state.quiz.q3 + '.' + value;
        this._showLeafOccupations(path);
      }

      UI.validateStep();
    },

    _buildQuestion(text, options, qKey) {
      var html = '<p class="quiz-prompt">' + escapeHtml(text) + '</p>';
      html += '<div class="quiz-options" role="radiogroup">';
      options.forEach(function (opt) {
        var selected = state.quiz[qKey] === opt.id ? ' selected' : '';
        html += '<button class="quiz-option' + selected + '" data-question="' + qKey + '" data-value="' + opt.id + '" type="button" role="radio" aria-checked="' + (selected ? 'true' : 'false') + '">';
        html += escapeHtml(opt.label);
        html += '</button>';
      });
      html += '</div>';
      return html;
    },

    reset() {
      state.quiz = { q1: null, q2: null, q3: null, q4: null };
      this.renderQ1();
    }
  };

  /* --------------------------------------------------------
     SEARCH ENGINE
     -------------------------------------------------------- */
  const SearchEngine = {
    highlightIndex: -1,

    search(query) {
      if (!query || query.length < CONFIG.search.minChars) return [];

      var q = query.toLowerCase();
      var words = q.split(/\s+/).filter(Boolean);

      var scored = [];
      state.searchArray.forEach(function (item) {
        var t = item.titleLower;
        var score = 0;

        // Exact substring match
        var pos = t.indexOf(q);
        if (pos !== -1) {
          if (pos === 0 || t[pos - 1] === ' ' || t[pos - 1] === ',') {
            score = 100;
          } else {
            score = 80;
          }
        } else if (words.length > 1) {
          var allPresent = words.every(function (w) { return t.indexOf(w) !== -1; });
          if (allPresent) {
            score = 60;
          } else {
            var count = words.filter(function (w) { return t.indexOf(w) !== -1; }).length;
            if (count > 0) {
              score = 30 * (count / words.length);
            }
          }
        }

        if (score > 0) {
          scored.push({ item: item, score: score, pos: pos >= 0 ? pos : 999 });
        }
      });

      scored.sort(function (a, b) {
        if (b.score !== a.score) return b.score - a.score;
        if (a.pos !== b.pos) return a.pos - b.pos;
        return a.item.title.localeCompare(b.item.title);
      });

      return scored.slice(0, CONFIG.search.maxResults).map(function (s) { return s.item; });
    },

    renderResults(results, query) {
      if (results.length === 0) {
        DOM.searchResults.innerHTML = '<li class="no-results">No occupations found for &ldquo;' + escapeHtml(query) + '&rdquo;</li>';
        DOM.searchResults.hidden = false;
        DOM.searchInput.setAttribute('aria-expanded', 'true');
        this.highlightIndex = -1;
        return;
      }

      var q = query.toLowerCase();
      var html = '';
      results.forEach(function (item, i) {
        var title = item.title;
        var lower = title.toLowerCase();
        var idx = lower.indexOf(q);
        var display;
        if (idx >= 0) {
          display = escapeHtml(title.substring(0, idx)) +
                    '<strong>' + escapeHtml(title.substring(idx, idx + query.length)) + '</strong>' +
                    escapeHtml(title.substring(idx + query.length));
        } else {
          display = escapeHtml(title);
        }
        html += '<li class="search-item" role="option" data-soc="' + item.soc + '" data-index="' + i + '">' + display + '</li>';
      });

      DOM.searchResults.innerHTML = html;
      DOM.searchResults.hidden = false;
      DOM.searchInput.setAttribute('aria-expanded', 'true');
      this.highlightIndex = -1;

      DOM.searchResults.querySelectorAll('.search-item').forEach(function (li) {
        li.addEventListener('click', function () {
          SearchEngine.selectResult(li.dataset.soc, li.textContent);
        });
      });
    },

    selectResult(soc, title) {
      state.selectedOccupation = soc;
      DOM.searchInput.value = title;
      DOM.searchResults.hidden = true;
      DOM.searchInput.setAttribute('aria-expanded', 'false');
      UI.showOccupationContext();
      UI.validateStep();
    },

    hideResults() {
      DOM.searchResults.hidden = true;
      DOM.searchInput.setAttribute('aria-expanded', 'false');
      this.highlightIndex = -1;
    },

    handleKeyboard(e) {
      var items = DOM.searchResults.querySelectorAll('.search-item');
      if (!items.length || DOM.searchResults.hidden) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        this.highlightIndex = Math.min(this.highlightIndex + 1, items.length - 1);
        this._updateHighlight(items);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        this.highlightIndex = Math.max(this.highlightIndex - 1, 0);
        this._updateHighlight(items);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (this.highlightIndex >= 0 && items[this.highlightIndex]) {
          var li = items[this.highlightIndex];
          this.selectResult(li.dataset.soc, li.textContent);
        }
      } else if (e.key === 'Escape') {
        this.hideResults();
      }
    },

    _updateHighlight(items) {
      items.forEach(function (li) { li.style.background = ''; });
      if (this.highlightIndex >= 0 && items[this.highlightIndex]) {
        items[this.highlightIndex].style.background = 'var(--color-primary-light)';
        items[this.highlightIndex].scrollIntoView({ block: 'nearest' });
      }
    }
  };

  /* --------------------------------------------------------
     METRO SELECTOR
     -------------------------------------------------------- */
  const MetroSelector = {
    highlightIndex: -1,

    search(query) {
      if (!query || query.length < CONFIG.search.minChars) return [];

      var q = query.toLowerCase();
      var words = q.split(/\s+/).filter(Boolean);

      var scored = [];
      state.cbsaArray.forEach(function (item) {
        var n = item.nameLower;
        var pos = n.indexOf(q);
        var score = 0;

        if (pos !== -1) {
          score = pos === 0 ? 100 : 80;
        } else {
          var allPresent = words.every(function (w) { return n.indexOf(w) !== -1; });
          if (allPresent) score = 60;
        }

        if (score > 0) {
          scored.push({ item: item, score: score, pos: pos >= 0 ? pos : 999 });
        }
      });

      scored.sort(function (a, b) {
        if (b.score !== a.score) return b.score - a.score;
        if (a.pos !== b.pos) return a.pos - b.pos;
        return b.item.employment - a.item.employment;
      });

      return scored.slice(0, CONFIG.search.maxResults).map(function (s) { return s.item; });
    },

    renderResults(results, query) {
      if (results.length === 0) {
        DOM.metroListbox.innerHTML = '<li class="no-results">No metro areas found</li>';
        DOM.metroListbox.hidden = false;
        DOM.inputMetro.setAttribute('aria-expanded', 'true');
        this.highlightIndex = -1;
        return;
      }

      var q = query.toLowerCase();
      var html = '';
      results.forEach(function (item, i) {
        var name = item.name;
        var lower = name.toLowerCase();
        var idx = lower.indexOf(q);
        var display;
        if (idx >= 0) {
          display = escapeHtml(name.substring(0, idx)) +
                    '<strong>' + escapeHtml(name.substring(idx, idx + query.length)) + '</strong>' +
                    escapeHtml(name.substring(idx + query.length));
        } else {
          display = escapeHtml(name);
        }
        html += '<li class="search-item" role="option" data-code="' + item.code + '" data-index="' + i + '">' + display + '</li>';
      });

      DOM.metroListbox.innerHTML = html;
      DOM.metroListbox.hidden = false;
      DOM.inputMetro.setAttribute('aria-expanded', 'true');
      this.highlightIndex = -1;

      DOM.metroListbox.querySelectorAll('.search-item').forEach(function (li) {
        li.addEventListener('click', function () {
          MetroSelector.selectResult(li.dataset.code);
        });
      });
    },

    selectResult(code) {
      var cbsa = state.density.cbsa_lookup[code];
      if (!cbsa) return;
      state.selectedMetro = code;
      state.selectedMetroName = cbsa.name;
      DOM.inputMetro.value = cbsa.name;
      DOM.metroListbox.hidden = true;
      DOM.inputMetro.setAttribute('aria-expanded', 'false');
      this.showLocationContext();
      UI.validateStep();
    },

    showLocationContext() {
      var code = state.selectedMetro;
      if (!code) return;

      var cbsa = state.density.cbsa_lookup[code];
      var occ = state.selectedOccupation ? state.benchmarks.occupations[state.selectedOccupation] : null;

      var html = '<h4>' + escapeHtml(cbsa.name) + '</h4>';
      html += '<dl class="context-stats">';
      html += '<div><dt>Total employment</dt><dd>' + formatNumber(cbsa.total_employment) + '</dd></div>';
      html += '<div><dt>Land area</dt><dd>' + formatNumber(Math.round(cbsa.land_area_sq_mi)) + ' sq mi</dd></div>';
      html += '<div><dt>Log employment density</dt><dd>' + cbsa.log_density.toFixed(2) + '</dd></div>';

      if (occ && occ.density_available) {
        var comparison = cbsa.log_density > occ.density_score
          ? 'denser than'
          : cbsa.log_density < occ.density_score
            ? 'less dense than'
            : 'similar to';
        html += '<div><dt>vs. your occupation\'s typical density</dt><dd>' + comparison + ' average (' + occ.density_score.toFixed(2) + ')</dd></div>';
      }

      html += '</dl>';
      DOM.locationContext.innerHTML = html;
      DOM.locationContext.hidden = false;
    },

    hideResults() {
      DOM.metroListbox.hidden = true;
      DOM.inputMetro.setAttribute('aria-expanded', 'false');
      this.highlightIndex = -1;
    },

    handleKeyboard(e) {
      var items = DOM.metroListbox.querySelectorAll('.search-item');
      if (!items.length || DOM.metroListbox.hidden) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        this.highlightIndex = Math.min(this.highlightIndex + 1, items.length - 1);
        this._updateHighlight(items);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        this.highlightIndex = Math.max(this.highlightIndex - 1, 0);
        this._updateHighlight(items);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (this.highlightIndex >= 0 && items[this.highlightIndex]) {
          this.selectResult(items[this.highlightIndex].dataset.code);
        }
      } else if (e.key === 'Escape') {
        this.hideResults();
      }
    },

    _updateHighlight(items) {
      items.forEach(function (li) { li.style.background = ''; });
      if (this.highlightIndex >= 0 && items[this.highlightIndex]) {
        items[this.highlightIndex].style.background = 'var(--color-primary-light)';
        items[this.highlightIndex].scrollIntoView({ block: 'nearest' });
      }
    }
  };

  /* --------------------------------------------------------
     COMPOSITE CALCULATOR
     -------------------------------------------------------- */
  const CompositeCalculator = {
    compute() {
      var soc = state.selectedOccupation;
      var occ = state.benchmarks.occupations[soc];
      if (!occ) return null;

      var months = state.financialRunway;
      var age = state.age;

      // 1. Transferability Z (from benchmarks)
      var transferZ = occ.transferability_z;

      // 2. Density Z (from benchmarks — may be missing)
      var densityZ = occ.density_available ? occ.density_z : null;

      // 3. Wealth Z — log-transform
      var wealthZ = (Math.log(months + 1) - Math.log(CONFIG.wealth.medianMonths + 1)) / CONFIG.wealth.scale;
      wealthZ = clamp(wealthZ, -CONFIG.zCap, CONFIG.zCap);

      // 4. Age Z — linear, younger = higher
      var ageZ = (CONFIG.ageCalc.center - age) / CONFIG.ageCalc.scale;
      ageZ = clamp(ageZ, -CONFIG.zCap, CONFIG.zCap);

      // Composite: average of available components
      var components = [transferZ, wealthZ, ageZ];
      if (densityZ !== null) components.push(densityZ);
      var compositeZ = components.reduce(function (a, b) { return a + b; }, 0) / components.length;

      // Convert to percentile
      var compositePercentile = normalCDF(compositeZ) * 100;

      // Component percentiles for display
      var transferPct = normalCDF(transferZ) * 100;
      var densityPct = densityZ !== null ? normalCDF(densityZ) * 100 : null;
      var wealthPct = normalCDF(wealthZ) * 100;
      var agePct = normalCDF(ageZ) * 100;

      var result = {
        soc: soc,
        occupation: occ,
        compositeZ: compositeZ,
        compositePercentile: compositePercentile,
        components: {
          transferability: { z: transferZ, percentile: transferPct, label: 'Skill transferability' },
          density:         { z: densityZ, percentile: densityPct, label: 'Geographic density', available: occ.density_available },
          wealth:          { z: wealthZ, percentile: wealthPct, label: 'Financial runway' },
          age:             { z: ageZ, percentile: agePct, label: 'Age factor' }
        },
        occupationComposite: occ.composite_score
      };

      state.results = result;
      return result;
    }
  };

  /* --------------------------------------------------------
     CHARTS
     -------------------------------------------------------- */
  const Charts = {
    createScatter() {
      if (!DOM.chartScatter) return;

      var occs = state.benchmarks.occupations;
      var data = [];
      Object.keys(occs).forEach(function (soc) {
        var o = occs[soc];
        if (o.ai_exposure != null && o.composite_score != null) {
          data.push({
            x: o.ai_exposure * 100,
            y: o.composite_score,
            r: Math.min(15, Math.max(2, Math.sqrt(o.employment_2024) * 0.5)),
            soc: soc,
            title: o.title,
            employment: o.employment_2024,
            aiExposure: o.ai_exposure
          });
        }
      });

      var ctx = DOM.chartScatter.getContext('2d');
      state.charts.scatter = new Chart(ctx, {
        type: 'bubble',
        data: {
          datasets: [{
            label: 'Occupations',
            data: data,
            backgroundColor: 'rgba(26, 86, 219, 0.15)',
            borderColor: 'rgba(26, 86, 219, 0.4)',
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          aspectRatio: 1.8,  // Fix 1: wider aspect ratio
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: function (tooltipItem) {
                  var d = tooltipItem.raw;
                  return d.title + ' — AI exposure: ' + (d.aiExposure * 100).toFixed(0) + '%, Adaptive capacity: ' + d.y.toFixed(0) + 'th pctl';
                }
              }
            }
          },
          scales: {
            x: {
              title: { display: true, text: 'AI Exposure Score (%)' },
              min: 0,
              max: 100
            },
            y: {
              title: { display: true, text: 'Adaptive Capacity (percentile)' },
              min: 0,
              max: 100
            }
          }
        }
      });

      // Fix 8: If occupation already selected, highlight it now
      if (state.selectedOccupation) {
        this.highlightOccupation(state.selectedOccupation);
      }
    },

    highlightOccupation(soc) {
      if (!state.charts.scatter) return;

      var chart = state.charts.scatter;
      var datasets = chart.data.datasets;

      // Remove previous highlight
      if (datasets.length > 1) {
        datasets.splice(1, 1);
      }

      if (!soc) {
        chart.update();
        return;
      }

      var occ = state.benchmarks.occupations[soc];
      if (!occ || occ.ai_exposure == null || occ.composite_score == null) {
        chart.update();
        return;
      }

      datasets.push({
        label: 'Your occupation',
        data: [{
          x: occ.ai_exposure * 100,
          y: occ.composite_score,
          r: 8
        }],
        backgroundColor: 'rgba(220, 38, 38, 0.8)',
        borderColor: '#dc2626',
        borderWidth: 2
      });

      chart.update();
    },

    // Fix 6: Custom plugin for bar labels
    _barLabelPlugin: {
      id: 'barLabels',
      afterDatasetsDraw: function (chart) {
        var ctx = chart.ctx;
        chart.data.datasets.forEach(function (dataset, datasetIndex) {
          var meta = chart.getDatasetMeta(datasetIndex);
          meta.data.forEach(function (bar, index) {
            var value = dataset.data[index];
            var x = bar.x;
            var y = bar.y;
            ctx.save();
            ctx.font = '600 12px system-ui, sans-serif';
            ctx.fillStyle = '#374151';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText(Math.round(value), x + 6, y);
            ctx.restore();
          });
        });
      }
    },

    createComponentBars(results) {
      if (!DOM.chartComponents) return;

      var comps = results.components;
      var labels = [];
      var values = [];
      var colors = [];

      var order = ['transferability', 'density', 'wealth', 'age'];

      order.forEach(function (key) {
        var c = comps[key];
        if (key === 'density' && !c.available) return;

        labels.push(c.label);
        var pct = c.percentile;
        values.push(pct);

        if (pct >= 60) colors.push(CONFIG.componentColors.high);
        else if (pct >= 40) colors.push(CONFIG.componentColors.mid);
        else colors.push(CONFIG.componentColors.low);
      });

      // Destroy old chart if exists
      if (state.charts.components) {
        state.charts.components.destroy();
        state.charts.components = null;
      }

      var ctx = DOM.chartComponents.getContext('2d');
      state.charts.components = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            data: values,
            backgroundColor: colors,
            borderRadius: 4,
            barPercentage: 0.6
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: true,
          aspectRatio: 2,
          layout: {
            padding: { right: 40 }  // Space for bar labels
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: function (tooltipItem) {
                  return ordinal(tooltipItem.parsed.x) + ' percentile';
                }
              }
            }
          },
          scales: {
            x: {
              min: 0,
              max: 100,
              title: { display: true, text: 'Percentile' }
            },
            y: {
              ticks: { font: { weight: '600' } }
            }
          }
        },
        plugins: [Charts._barLabelPlugin]
      });
    }
  };

  /* --------------------------------------------------------
     RESULTS RENDERER
     -------------------------------------------------------- */
  const ResultsRenderer = {
    render(results) {
      this._renderHeadline(results);
      Charts.createComponentBars(results);
      this._renderCallouts(results);
      this._renderRoadmap(results);
      this._renderNeighbors(results);
    },

    // C1: Descriptive tier + percentile
    _renderHeadline(r) {
      var pct = Math.round(r.compositePercentile);
      var occ = r.occupation;
      var tier = getScoreTier(pct);

      var levelText = {
        'Strong Position':  'You\u2019re in a strong position. If your work changes due to AI, you have real options for what comes next.',
        'Solid Foundation': 'You\u2019ve got a solid foundation. With some deliberate effort, you could improve your options.',
        'Room to Build':    'You\u2019re not starting from zero, there are real things working in your favor here. But there are also areas where attention now could pay off later.',
        'Early Stages':     'This is a hard position to adapt from. There are strategies here for making the most of your position, but your starting options are probably limited.'
      };

      var html = '<div class="score-display">';
      html += '<div class="score-circle" style="--score: ' + pct + '">';
      html += '<span class="score-number">' + escapeHtml(tier.label) + '</span>';
      html += '<span class="score-suffix">' + pct + ' of 100</span>';
      html += '</div>';
      html += '<div class="score-context">';
      html += '<h3>' + escapeHtml(occ.title) + '</h3>';
      html += '<p>' + (levelText[tier.label] || '') + '</p>';

      // Comparison to occupation benchmark
      if (occ.composite_score != null) {
        var occPct = Math.round(occ.composite_score);
        var diff = pct - occPct;
        var comparison;
        if (Math.abs(diff) <= 5) {
          comparison = 'roughly in line with';
        } else if (diff > 0) {
          comparison = Math.abs(diff) + ' points above';
        } else {
          comparison = Math.abs(diff) + ' points below';
        }
        html += '<p style="font-size: var(--text-sm); color: var(--color-text-secondary);">' +
                'Your personal score (' + ordinal(pct) + ' percentile) is ' + comparison + ' the occupation-level benchmark (' + ordinal(occPct) + ' percentile), which accounts for skill transferability and geographic density only.</p>';
      }

      // AI exposure note
      if (occ.ai_exposure != null) {
        var expPct = Math.round(occ.ai_exposure * 100);
        html += '<div class="ai-exposure-note">';
        html += '<strong>AI exposure:</strong> ' + expPct + '% (';
        if (expPct >= 70) html += 'high';
        else if (expPct >= 40) html += 'moderate';
        else html += 'lower';
        html += ' \u2014 ' + ordinal(occ.ai_exposure_percentile) + ' percentile across occupations)';
        html += '</div>';
      }

      html += '</div></div>';
      DOM.resultsHeadline.innerHTML = html;
    },

    _renderCallouts(r) {
      var comps = r.components;
      var available = [];
      ['transferability', 'density', 'wealth', 'age'].forEach(function (key) {
        var c = comps[key];
        if (key === 'density' && !c.available) return;
        available.push({ key: key, label: c.label, percentile: c.percentile });
      });

      available.sort(function (a, b) { return b.percentile - a.percentile; });
      var strongest = available[0];
      var weakest = available[available.length - 1];

      // C8: Softer language when age is weakest
      var weakLabel = weakest.key === 'age' ? 'Factor to plan around' : 'Area to address';

      var strongText = {
        transferability: 'The transferable skills your work builds are useful in a lot of places, not just your current role. You\u2019re more likely to have a lot of options if things shift.',
        density:         'You\u2019re in a large labor market \u2014 lots of employers hiring for lots of roles. That means more chances to find a good match, not just any match.',
        wealth:          'Your financial buffer gives you more time during change. That means you can wait for the right next thing rather than needing to grab for the first next thing. That difference matters a lot.',
        age:             'Being young in the labor market means you\u2019ve got time on your side. Retraining, relocating, pivoting \u2014 all of these are easier when you\u2019re earlier in your career.'
      };

      var weakText = {
        transferability: 'Your skills run deep, which is an asset in your current role, but specialization can limit options if your speciality is threatened by AI. If you needed to make a lateral move, you might benefit from more transferable skills in your toolkit.',
        density:         'Your local market is thinner than average. Fewer employers, and fewer open roles at any given time. That doesn\u2019t mean you\u2019re stuck, but it means your search radius matters more. Remote-eligible roles and a willingness to relocate can change this number.',
        wealth:          'With less financial runway, the clock ticks louder during a search. That pressure can push people toward roles that are available rather than roles that are right. That mismatch can persist for years.',
        age:             'The research is honest about this: career transitions get harder with age. Not impossible. The most efficient path is building from what you already know rather than starting fresh in a new field. Lean into age advantages, like your network and your perspective.'
      };

      var html = '<div class="callouts-row">';
      html += '<div class="callout callout-strong">';
      html += '<h4>Strongest dimension</h4>';
      html += '<p><strong>' + strongest.label + '</strong> \u2014 ' + ordinal(strongest.percentile) + ' percentile</p>';
      html += '<p style="font-size: var(--text-sm); color: var(--color-text-secondary);">' + (strongText[strongest.key] || '') + '</p>';
      html += '</div>';

      html += '<div class="callout callout-weak">';
      html += '<h4>' + weakLabel + '</h4>';
      html += '<p><strong>' + weakest.label + '</strong> \u2014 ' + ordinal(weakest.percentile) + ' percentile</p>';
      html += '<p style="font-size: var(--text-sm); color: var(--color-text-secondary);">' + (weakText[weakest.key] || '') + '</p>';
      html += '</div>';
      html += '</div>';

      DOM.resultsCallouts.innerHTML = html;
    },

    _renderRoadmap(r) {
      var comps = r.components;
      var items = [];

      ['transferability', 'density', 'wealth', 'age'].forEach(function (key) {
        var c = comps[key];
        if (key === 'density' && !c.available) return;
        items.push({ key: key, label: c.label, percentile: c.percentile });
      });

      items.sort(function (a, b) { return a.percentile - b.percentile; });
      var roadmapItems = items.slice(0, Math.min(3, items.length));

      // C8: Softer age roadmap text
      var roadmapText = {
        transferability: {
          title: 'Build transferable skills',
          text: 'Look at the growing occupations on your neighbor list. What do they have in common that your current role doesn\u2019t? Those overlapping skills (the ones that show up across multiple neighbors) are where your learning investment has the widest payoff.'
        },
        density: {
          title: 'Consider location strategy',
          text: 'A denser labor market means more shots on goal. That could mean physically relocating, or it could mean making yourself eligible for remote roles, which effectively puts you in every dense market at once.'
        },
        wealth: {
          title: 'Strengthen financial buffer',
          text: 'Every additional month of runway changes the math on your search. It\u2019s the difference between \u2018I need a job now\u2019 and \u2018I need the right job.\u2019 Even modest, steady progress here shifts which category you\u2019re in.'
        },
        age: {
          title: 'Plan around your career timeline',
          text: 'You don\u2019t need to reinvent yourself, you need to extend yourself. The moves with the best return from here are the ones where 80% of what you already know still applies, and the 20% that\u2019s new opens a door that didn\u2019t exist before.'
        }
      };

      var html = '<ol class="roadmap-list">';
      roadmapItems.forEach(function (item) {
        var info = roadmapText[item.key];
        html += '<li class="roadmap-item">';
        html += '<h4>' + info.title + '</h4>';
        html += '<p>' + info.text + '</p>';
        html += '</li>';
      });
      html += '</ol>';

      DOM.roadmapContent.innerHTML = html;
    },

    // C2: Add AI exposure to growing neighbors
    _renderNeighbors(r) {
      var neighbors = r.occupation.growing_neighbors;
      if (!neighbors || neighbors.length === 0) {
        DOM.neighborsContent.innerHTML = '<p class="hint">No growing occupations with similar skill profiles found.</p>';
        return;
      }

      var html = '<div class="neighbors-list">';
      neighbors.forEach(function (n) {
        // Look up AI exposure from benchmarks
        var neighborOcc = state.benchmarks.occupations[n.soc];
        var aiExp = neighborOcc && neighborOcc.ai_exposure != null
          ? Math.round(neighborOcc.ai_exposure * 100)
          : null;

        // Color-code AI exposure
        var aiCls = '';
        if (aiExp !== null) {
          if (aiExp >= 70) aiCls = ' style="color: var(--color-danger);"';
          else if (aiExp >= 40) aiCls = ' style="color: var(--color-warning);"';
          else aiCls = ' style="color: var(--color-success);"';
        }

        html += '<div class="neighbor-card">';
        html += '<h4>' + escapeHtml(n.title) + '</h4>';
        html += '<div class="neighbor-stats">';
        html += '<span>Similarity: ' + (n.similarity * 100).toFixed(0) + '%</span>';
        html += '<span>Growth: ' + formatGrowth(n.growth_rate) + '</span>';
        if (aiExp !== null) {
          html += '<span' + aiCls + '>AI exposure: ' + aiExp + '%</span>';
        }
        html += '<span>Employment: ' + formatNumber(Math.round(n.employment_2024 * 1000)) + '</span>';
        html += '</div>';
        html += '</div>';
      });
      html += '</div>';

      DOM.neighborsContent.innerHTML = html;
    }
  };

  /* --------------------------------------------------------
     UI CONTROLLER
     -------------------------------------------------------- */
  const UI = {
    init() {
      this.bindEvents();
      QuizEngine.init();
    },

    bindEvents() {
      // Hero buttons
      DOM.btnStart.addEventListener('click', function () {
        DOM.assessment.hidden = false;
        DOM.assessment.scrollIntoView({ behavior: 'smooth' });
      });

      DOM.btnLearn.addEventListener('click', function () {
        var isHidden = DOM.explainer.hidden;
        DOM.explainer.hidden = !isHidden;
        if (!isHidden) return;
        DOM.explainer.scrollIntoView({ behavior: 'smooth' });
        if (!state.charts.scatter) {
          Charts.createScatter();
        }
      });

      // Tab toggle
      DOM.tabQuiz.addEventListener('click', function () {
        DOM.tabQuiz.classList.add('active');
        DOM.tabSearch.classList.remove('active');
        DOM.tabQuiz.setAttribute('aria-selected', 'true');
        DOM.tabSearch.setAttribute('aria-selected', 'false');
        DOM.panelQuiz.hidden = false;
        DOM.panelSearch.hidden = true;
      });

      DOM.tabSearch.addEventListener('click', function () {
        DOM.tabSearch.classList.add('active');
        DOM.tabQuiz.classList.remove('active');
        DOM.tabSearch.setAttribute('aria-selected', 'true');
        DOM.tabQuiz.setAttribute('aria-selected', 'false');
        DOM.panelSearch.hidden = false;
        DOM.panelQuiz.hidden = true;
      });

      // Quiz delegation
      var quizContainer = DOM.panelQuiz;
      quizContainer.addEventListener('click', function (e) {
        var btn = e.target.closest('.quiz-option');
        if (!btn) return;
        var q = btn.dataset.question;
        var v = btn.dataset.value;
        if (q && v) {
          btn.parentElement.querySelectorAll('.quiz-option').forEach(function (b) {
            b.classList.remove('selected');
            b.setAttribute('aria-checked', 'false');
          });
          btn.classList.add('selected');
          btn.setAttribute('aria-checked', 'true');
          QuizEngine.handleSelection(q, v);
        }
      });

      // Search input
      var debouncedSearch = debounce(function () {
        var q = DOM.searchInput.value.trim();
        if (q.length < CONFIG.search.minChars) {
          SearchEngine.hideResults();
          return;
        }
        var results = SearchEngine.search(q);
        SearchEngine.renderResults(results, q);
      }, CONFIG.search.debounceMs);

      DOM.searchInput.addEventListener('input', function () {
        state.selectedOccupation = null;
        DOM.occContext.hidden = true;
        UI.validateStep();
        debouncedSearch();
      });

      DOM.searchInput.addEventListener('keydown', function (e) {
        SearchEngine.handleKeyboard(e);
      });

      // Close dropdowns when clicking outside
      document.addEventListener('click', function (e) {
        if (!e.target.closest('#panel-search')) {
          SearchEngine.hideResults();
        }
        if (!e.target.closest('#step-location')) {
          MetroSelector.hideResults();
        }
      });

      // Metro input
      var debouncedMetro = debounce(function () {
        var q = DOM.inputMetro.value.trim();
        if (q.length < CONFIG.search.minChars) {
          MetroSelector.hideResults();
          return;
        }
        var results = MetroSelector.search(q);
        MetroSelector.renderResults(results, q);
      }, CONFIG.search.debounceMs);

      DOM.inputMetro.addEventListener('input', function () {
        state.selectedMetro = null;
        state.selectedMetroName = null;
        DOM.locationContext.hidden = true;
        UI.validateStep();
        debouncedMetro();
      });

      DOM.inputMetro.addEventListener('keydown', function (e) {
        MetroSelector.handleKeyboard(e);
      });

      // Fix 4: Financial runway input — handle empty string explicitly
      DOM.inputRunway.addEventListener('input', function () {
        var raw = DOM.inputRunway.value.trim();
        if (raw === '') {
          state.financialRunway = null;
          DOM.runwayIndicator.hidden = true;
          UI.validateStep();
          return;
        }
        var val = parseInt(raw, 10);
        if (isNaN(val) || val < 0) {
          state.financialRunway = null;
          DOM.runwayIndicator.hidden = true;
        } else {
          state.financialRunway = val;
          UI.showRunwayIndicator(val);
        }
        UI.validateStep();
      });

      // C5: Age input — show out-of-range validation
      DOM.inputAge.addEventListener('input', function () {
        var raw = DOM.inputAge.value.trim();
        if (raw === '') {
          state.age = null;
          DOM.ageContext.hidden = true;
          UI.validateStep();
          return;
        }
        var val = parseInt(raw, 10);
        if (isNaN(val)) {
          state.age = null;
          DOM.ageContext.hidden = true;
        } else if (val < 16 || val > 85) {
          state.age = null;
          DOM.ageContext.innerHTML = '<p style="color: var(--color-danger); font-weight: 600;">Please enter an age between 16 and 85</p>';
          DOM.ageContext.hidden = false;
        } else {
          state.age = val;
          UI.showAgeContext(val);
        }
        UI.validateStep();
      });

      // Step navigation
      DOM.btnNext.addEventListener('click', function () {
        UI.goToStep(state.currentStep + 1);
      });

      DOM.btnPrev.addEventListener('click', function () {
        UI.goToStep(state.currentStep - 1);
      });

      DOM.btnResults.addEventListener('click', function () {
        UI.showResults();
      });

      // Restart — both from results and from step nav
      DOM.btnRestart.addEventListener('click', function () {
        UI.restart();
      });

      // C7: Start over button in step nav
      if (DOM.btnStartOver) {
        DOM.btnStartOver.addEventListener('click', function (e) {
          e.preventDefault();
          UI.restart();
        });
      }
    },

    // Fix 3: Scroll occupation context into view
    showOccupationContext() {
      var soc = state.selectedOccupation;
      if (!soc) return;

      var occ = state.benchmarks.occupations[soc];
      if (!occ) return;

      var html = '<h4>' + escapeHtml(occ.title) + '</h4>';
      html += '<dl class="context-stats">';
      html += '<div><dt>AI exposure</dt><dd>' + (occ.ai_exposure != null ? (occ.ai_exposure * 100).toFixed(0) + '%' : '—') + '</dd></div>';
      html += '<div><dt>Skill transferability</dt><dd>' + ordinal(occ.transferability_percentile) + ' percentile</dd></div>';

      if (occ.composite_score != null) {
        html += '<div><dt>Occupation adaptive capacity</dt><dd>' + ordinal(occ.composite_score) + ' percentile</dd></div>';
      }

      html += '<div><dt>Median annual wage</dt><dd>' + formatWage(occ.median_annual_wage, occ.wage_top_coded) + '</dd></div>';
      html += '<div><dt>Projected growth (2024–2034)</dt><dd>' + formatGrowth(occ.growth_rate) + '</dd></div>';
      html += '<div><dt>Employment (2024)</dt><dd>' + formatNumber(Math.round(occ.employment_2024 * 1000)) + '</dd></div>';
      html += '</dl>';

      DOM.occContext.innerHTML = html;
      DOM.occContext.hidden = false;

      // Fix 3: Scroll context card into view
      setTimeout(function () {
        DOM.occContext.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 50);

      // Highlight on scatter chart if visible
      Charts.highlightOccupation(soc);
    },

    showRunwayIndicator(months) {
      var band = CONFIG.runway.bands.find(function (b) { return months < b.max; });
      if (!band) return;

      DOM.runwayIndicator.className = 'indicator ' + band.cls;
      DOM.runwayIndicator.textContent = band.label;
      DOM.runwayIndicator.hidden = false;
    },

    showAgeContext(age) {
      var band = CONFIG.age.bands.find(function (b) { return age <= b.max; });
      if (!band) return;

      DOM.ageContext.innerHTML = '<p><strong>' + band.label + '</strong></p>';
      DOM.ageContext.hidden = false;
    },

    validateStep() {
      var valid = false;
      switch (state.currentStep) {
        case 1: valid = state.selectedOccupation != null; break;
        case 2: valid = state.financialRunway != null && state.financialRunway >= 0; break;
        case 3: valid = state.selectedMetro != null; break;
        case 4: valid = state.age != null && state.age >= 16 && state.age <= 85; break;
      }

      if (state.currentStep === 4) {
        DOM.btnResults.disabled = !valid;
      } else {
        DOM.btnNext.disabled = !valid;
      }
    },

    goToStep(step) {
      if (step < 1 || step > 4) return;

      state.currentStep = step;

      // Show/hide steps
      var steps = [DOM.stepOccupation, DOM.stepFinancial, DOM.stepLocation, DOM.stepAge];
      steps.forEach(function (el, i) {
        el.hidden = (i + 1) !== step;
      });

      // Update progress
      var pct = (step / 4) * 100;
      DOM.progressFill.style.width = pct + '%';
      DOM.progressLabel.textContent = 'Step ' + step + ' of 4';

      // Update ARIA
      var progressBar = DOM.progressFill.parentElement;
      progressBar.setAttribute('aria-valuenow', step);

      // Show/hide nav buttons
      DOM.btnPrev.hidden = step === 1;
      DOM.btnNext.hidden = step === 4;
      DOM.btnResults.hidden = step !== 4;

      // C7: Show start over on steps 2+
      if (DOM.btnStartOver) {
        DOM.btnStartOver.hidden = step === 1;
      }

      this.validateStep();

      // Scroll to top of assessment
      DOM.assessment.scrollIntoView({ behavior: 'smooth' });
    },

    showResults() {
      var results = CompositeCalculator.compute();
      if (!results) return;

      DOM.assessment.hidden = true;
      DOM.results.hidden = false;
      DOM.results.scrollIntoView({ behavior: 'smooth' });

      ResultsRenderer.render(results);
    },

    restart() {
      // Reset state
      state.selectedOccupation = null;
      state.financialRunway = null;
      state.selectedMetro = null;
      state.selectedMetroName = null;
      state.age = null;
      state.results = null;
      state.currentStep = 1;

      // Reset inputs
      DOM.inputRunway.value = '';
      DOM.inputAge.value = '';
      DOM.inputMetro.value = '';
      DOM.searchInput.value = '';

      // Hide context cards
      DOM.occContext.hidden = true;
      DOM.runwayIndicator.hidden = true;
      DOM.locationContext.hidden = true;
      DOM.ageContext.hidden = true;

      // Reset quiz
      QuizEngine.reset();

      // Hide results, show assessment
      DOM.results.hidden = true;
      DOM.assessment.hidden = false;

      // Reset to step 1
      this.goToStep(1);

      // Switch back to quiz tab
      DOM.tabQuiz.classList.add('active');
      DOM.tabSearch.classList.remove('active');
      DOM.tabQuiz.setAttribute('aria-selected', 'true');
      DOM.tabSearch.setAttribute('aria-selected', 'false');
      DOM.panelQuiz.hidden = false;
      DOM.panelSearch.hidden = true;

      // Fix 9: Remove scatter highlight and destroy component chart
      if (state.charts.scatter) {
        Charts.highlightOccupation(null);
      }
      if (state.charts.components) {
        state.charts.components.destroy();
        state.charts.components = null;
      }

      // Scroll to hero
      DOM.hero.scrollIntoView({ behavior: 'smooth' });
    }
  };

  /* --------------------------------------------------------
     APP INIT
     -------------------------------------------------------- */
  const App = {
    async init() {
      try {
        await DataLoader.loadAll();
        DOM.overlay.hidden = true;
        UI.init();
      } catch (err) {
        console.error('Failed to load data:', err);
        DOM.overlay.innerHTML =
          '<p class="error">Failed to load assessment data</p>' +
          '<p class="error-detail">' + escapeHtml(err.message) + '</p>' +
          '<p class="error-detail">Please make sure you\'re serving this from a web server, not opening the HTML file directly.</p>';
      }
    }
  };

  // Boot
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { App.init(); });
  } else {
    App.init();
  }

})();
