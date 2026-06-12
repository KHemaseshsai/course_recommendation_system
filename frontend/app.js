const backendCandidates = [];
if (window.location.protocol.startsWith("http")) {
    backendCandidates.push(window.location.origin);
}
backendCandidates.push("http://127.0.0.1:8000", "http://localhost:8000");

const API_BASE_URLS = [...new Set(backendCandidates)];
let activeApiBaseUrl = API_BASE_URLS[0];
const userId = getOrCreateUserId();
const sessionId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;

const defaults = {
    goal: "",
    level: "Beginner",
    learning_style: "Balanced",
    time_commitment: "5-7 hrs/week",
    budget_preference: "Any",
    delivery_preference: "Any",
    certificate_preference: "Any",
    preferred_platform: "Any",
    limit: 6,
};

const fallbackMetadata = {
    levels: ["Beginner", "Intermediate", "Advanced", "All Levels"],
    learning_styles: ["Hands-on", "Structured", "Exam-focused", "Fast-track", "Balanced"],
    time_commitments: ["Flexible", "Weekend only", "5-7 hrs/week", "8-12 hrs/week", "12+ hrs/week"],
    budget_options: ["Any", "Free", "Paid"],
    delivery_options: ["Any", "Project-based", "Theory-based"],
    certificate_options: ["Any", "Yes", "No"],
    platforms: ["Any", "Coursera", "Future Learn", "Simplilearn", "Udacity"],
};

const platformColors = ["#ff6b35", "#0b8f87", "#ffb703", "#7b8cff", "#ff8fab", "#3a86ff"];
const form = document.getElementById("recommendationForm");
const statusBanner = document.getElementById("statusBanner");
const resultsSection = document.getElementById("resultsSection");
const tabButtons = Array.from(document.querySelectorAll(".tab-button"));
const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));
const limitInput = document.getElementById("limit");
const limitValue = document.getElementById("limitValue");
const submitButton = document.getElementById("submitButton");
const activeFilters = document.getElementById("activeFilters");
const formFieldIds = ["goal", "level", "learning_style", "time_commitment", "budget_preference", "delivery_preference", "certificate_preference", "preferred_platform", "limit"];

function getOrCreateUserId() {
    const storageKey = "course_recommender_user_id";
    const existing = localStorage.getItem(storageKey);
    if (existing) {
        return existing;
    }
    const generated = crypto.randomUUID ? crypto.randomUUID() : `user-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    localStorage.setItem(storageKey, generated);
    return generated;
}

function setSelectLoadingState(id, message) {
    const select = document.getElementById(id);
    if (!select) {
        return;
    }
    select.innerHTML = `<option value="">${message}</option>`;
    select.disabled = true;
}

function populateSelect(id, options, fallback) {
    const select = document.getElementById(id);
    if (!select) {
        return;
    }

    select.innerHTML = "";
    select.disabled = false;

    options.forEach((option) => {
        const optionElement = document.createElement("option");
        optionElement.value = option;
        optionElement.textContent = option;
        select.appendChild(optionElement);
    });

    select.value = options.includes(fallback) ? fallback : options[0] || "";
}

function applyMetadata(metadata) {
    populateSelect("level", metadata.levels || fallbackMetadata.levels, defaults.level);
    populateSelect("learning_style", metadata.learning_styles || fallbackMetadata.learning_styles, defaults.learning_style);
    populateSelect("delivery_preference", metadata.delivery_options || fallbackMetadata.delivery_options, defaults.delivery_preference);
    populateSelect("certificate_preference", metadata.certificate_options || fallbackMetadata.certificate_options, defaults.certificate_preference);
    populateSelect("time_commitment", metadata.time_commitments || fallbackMetadata.time_commitments, defaults.time_commitment);
    populateSelect("budget_preference", metadata.budget_options || fallbackMetadata.budget_options, defaults.budget_preference);
    populateSelect("preferred_platform", metadata.platforms || fallbackMetadata.platforms, defaults.preferred_platform);
}

function setFormBusy(isBusy) {
    formFieldIds.forEach((id) => {
        const field = document.getElementById(id);
        if (field) {
            field.disabled = isBusy;
        }
    });

    submitButton.disabled = isBusy;
    submitButton.textContent = isBusy ? "Scoring Matches..." : "Generate Personalized Recommendations";
}

function getPayload() {
    const formData = new FormData(form);
    return {
        goal: String(formData.get("goal") || "").trim(),
        level: String(formData.get("level") || defaults.level),
        learning_style: String(formData.get("learning_style") || defaults.learning_style),
        time_commitment: String(formData.get("time_commitment") || defaults.time_commitment),
        completion_months: "Any",
        budget_preference: String(formData.get("budget_preference") || defaults.budget_preference),
        delivery_preference: String(formData.get("delivery_preference") || defaults.delivery_preference),
        certificate_preference: String(formData.get("certificate_preference") || defaults.certificate_preference),
        preferred_platform: String(formData.get("preferred_platform") || defaults.preferred_platform),
        preferred_language: "Any",
        category: "Any",
        limit: Number(formData.get("limit") || defaults.limit),
        user_id: userId,
    };
}

async function recordInteraction(course, eventType) {
    const params = new URLSearchParams({
        event_type: eventType,
        course: course.course || "",
        course_id: course.course_id || "",
        category: course.category || "",
        sub_category: course.sub_category || "",
        platform: course.platform || "",
        level: course.level || course.difficulty || "",
        session_id: sessionId,
        source: "frontend",
    });
    try {
        await fetch(`${activeApiBaseUrl}/users/${encodeURIComponent(userId)}/events?${params.toString()}`, {
            method: "POST",
            keepalive: true,
        });
    } catch (error) {
        console.debug("Interaction telemetry unavailable", error);
    }
}

function setStatus(message, kind) {
    statusBanner.textContent = message;
    statusBanner.className = `status-banner ${kind}`;
}

function clearStatus() {
    statusBanner.textContent = "";
    statusBanner.className = "status-banner hidden";
}

function truncate(text, maxLength) {
    if (!text) {
        return "";
    }
    return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function activateTab(tabName) {
    tabButtons.forEach((button) => {
        button.classList.toggle("active", button.dataset.tab === tabName);
    });
    tabPanels.forEach((panel) => {
        panel.classList.toggle("active", panel.id === tabName);
    });
}

function renderFilterPills(payload, recommendationCount) {
    const pills = [
        `Goal: ${payload.goal || "Not provided"}`,
        `Level: ${payload.level}`,
        `Learning style: ${payload.learning_style}`,
        `Delivery: ${payload.delivery_preference}`,
        `Certificate: ${payload.certificate_preference}`,
        `Time: ${payload.time_commitment}`,
        `Budget: ${payload.budget_preference}`,
        `Platform: ${payload.preferred_platform}`,
        `Results: ${recommendationCount}`,
    ];

    activeFilters.innerHTML = pills
        .map((item) => `<span class="profile-pill">${escapeHtml(item)}</span>`)
        .join("");
}

function renderChart(recommendations) {
    const chartBars = document.getElementById("chartBars");
    chartBars.innerHTML = "";

    const maxScore = Math.max(...recommendations.map((item) => Number(item.score) || 0), 0.55);
    const platforms = [...new Set(recommendations.map((item) => item.platform))];
    const colorMap = new Map(platforms.map((platform, index) => [platform, platformColors[index % platformColors.length]]));

    recommendations.forEach((item) => {
        const row = document.createElement("div");
        row.className = "chart-row";
        const width = Math.max(16, ((Number(item.score) || 0) / maxScore) * 100);
        const shortCourse = truncate(item.course, 34);
        const color = colorMap.get(item.platform) || platformColors[0];

        row.innerHTML = `
            <div class="chart-label" title="${escapeHtml(item.course)}">${escapeHtml(shortCourse)}</div>
            <div class="chart-track">
                <div class="chart-bar" style="width: ${width}%; background: ${color};">${escapeHtml(item.platform || "Platform")}</div>
            </div>
            <div class="chart-score">${Number(item.score || 0).toFixed(2)}</div>
        `;
        chartBars.appendChild(row);
    });
}

function renderCourseCards(recommendations) {
    const courseGrid = document.getElementById("courseGrid");
    courseGrid.innerHTML = "";

    recommendations.forEach((course, index) => {
        const skillsPreview = String(course.skills || "")
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean)
            .slice(0, 4)
            .join(", ") || "Not specified";
        const comparison = course.comparison || {};
        const progress = Math.max(0, Math.min(100, Math.round((Number(course.score) || 0) * 100)));
        const hasUrl = Boolean(course.url);

        const card = document.createElement("article");
        card.className = "course-card";
        card.innerHTML = `
            <div class="card-kicker">Rank #${index + 1}</div>
            <h3>${escapeHtml(course.course)}</h3>
            <div class="course-meta">${escapeHtml(course.difficulty)} | ${escapeHtml(course.platform)} | ${escapeHtml(course.time_estimate)}</div>
            <p class="course-body"><strong>Estimated completion:</strong> ${escapeHtml(course.estimated_completion || "Depends on pace")}</p>
            <p class="course-body"><strong>Budget:</strong> ${escapeHtml(course.budget_type || "Unknown")} | <strong>Format:</strong> ${escapeHtml(course.delivery_mode || "Unknown")} | <strong>Certificate:</strong> ${course.certificate_available ? "Yes" : "No"}</p>
            <p class="course-body"><strong>Why this fits:</strong> ${escapeHtml(course.reason)}</p>
            <p class="course-body">${escapeHtml(course.description)}</p>
            <p class="course-body"><strong>Peer feedback:</strong> ${escapeHtml(course.peer_feedback_summary)}</p>
            <p class="course-body"><strong>Score:</strong> ${escapeHtml(course.score)} | <strong>Rating:</strong> ${escapeHtml(course.rating)}</p>
            <div class="tag-row"><strong>Skills:</strong> ${escapeHtml(skillsPreview)}</div>
            <div class="tag-row"><strong>Fit breakdown:</strong> Goal ${escapeHtml(comparison["Goal alignment"] ?? 0)} | Level ${escapeHtml(comparison["Level fit"] ?? 0)} | Time ${escapeHtml(comparison["Time fit"] ?? 0)} | Finish ${escapeHtml(comparison["Completion fit"] ?? 0)} | Budget ${escapeHtml(comparison["Budget fit"] ?? 0)} | Delivery ${escapeHtml(comparison["Delivery fit"] ?? 0)} | Certificate ${escapeHtml(comparison["Certificate fit"] ?? 0)}</div>
            <div class="course-actions">
                ${hasUrl ? `<a class="cta-link course-open" href="${escapeHtml(course.url)}" target="_blank" rel="noreferrer">Open Course</a>` : `<span class="cta-link disabled-link">Link unavailable</span>`}
                <button type="button" class="cta-link course-event" data-event="save">Save</button>
                <button type="button" class="cta-link course-event" data-event="enroll">Enrolled</button>
                <button type="button" class="cta-link course-event" data-event="complete">Completed</button>
            </div>
            <div class="progress-shell"><div class="progress-fill" style="width: ${progress}%;"></div></div>
        `;
        const openLink = card.querySelector(".course-open");
        if (openLink) {
            openLink.addEventListener("click", () => recordInteraction(course, "click"));
        }
        card.querySelectorAll(".course-event").forEach((button) => {
            button.addEventListener("click", async () => {
                await recordInteraction(course, button.dataset.event);
                button.textContent = "Recorded";
                button.disabled = true;
            });
        });
        courseGrid.appendChild(card);
        recordInteraction(course, "view");
    });
}

function renderCompare(recommendations, comparisonRows) {
    const compareCards = document.getElementById("compareCards");
    const comparisonTableWrap = document.getElementById("comparisonTableWrap");
    compareCards.innerHTML = "";
    comparisonTableWrap.innerHTML = "";

    recommendations.slice(0, 3).forEach((course) => {
        const card = document.createElement("article");
        card.className = "compare-card";
        card.innerHTML = `
            <h3>${escapeHtml(course.course)}</h3>
            <p class="course-meta">${escapeHtml(course.platform)} | ${escapeHtml(course.difficulty)}</p>
            <p class="course-body"><strong>Time:</strong> ${escapeHtml(course.time_estimate)}</p>
            <p class="course-body"><strong>Estimated completion:</strong> ${escapeHtml(course.estimated_completion || "Depends on pace")}</p>
            <p class="course-body"><strong>Budget:</strong> ${escapeHtml(course.budget_type || "Unknown")}</p>
            <p class="course-body"><strong>Format:</strong> ${escapeHtml(course.delivery_mode || "Unknown")}</p>
            <p class="course-body"><strong>Certificate:</strong> ${course.certificate_available ? "Yes" : "No"}</p>
            <p class="course-body"><strong>Rating:</strong> ${escapeHtml(course.rating)}</p>
            <p class="course-body"><strong>Why it fits:</strong> ${escapeHtml(course.reason)}</p>
        `;
        compareCards.appendChild(card);
    });

    if (!comparisonRows.length) {
        return;
    }

    const columns = Object.keys(comparisonRows[0]);
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const tbody = document.createElement("tbody");

    thead.innerHTML = `<tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>`;
    tbody.innerHTML = comparisonRows.map((row) => `
        <tr>
            ${columns.map((column) => `<td>${escapeHtml(row[column] ?? "")}</td>`).join("")}
        </tr>
    `).join("");

    table.appendChild(thead);
    table.appendChild(tbody);
    comparisonTableWrap.appendChild(table);
}

function renderRefinementPrompts(prompts) {
    const promptList = document.getElementById("promptList");
    promptList.innerHTML = "";

    prompts.forEach((prompt) => {
        const chip = document.createElement("div");
        chip.className = "prompt-chip";
        chip.textContent = prompt;
        promptList.appendChild(chip);
    });
}

function renderResults(data, payload) {
    const recommendations = data.recommendations || [];
    if (!recommendations.length) {
        setStatus("No recommendations matched the current filters. Try broadening budget, certificate, platform, language, time, or completion target.", "warning");
        resultsSection.classList.add("hidden");
        return;
    }

    document.getElementById("matchesFound").textContent = data.count ?? recommendations.length;
    document.getElementById("preferredProfile").textContent = `${payload.level} / ${payload.learning_style}`;
    document.getElementById("availabilityFocus").textContent = payload.time_commitment;
    document.getElementById("coachMessage").textContent = data.coach_message || "";
    renderFilterPills(payload, recommendations.length);

    const best = recommendations[0];
    document.getElementById("bestCourseTitle").textContent = best.course || "";
    document.getElementById("bestCourseReason").textContent = best.reason || "";
    document.getElementById("bestCourseMeta").innerHTML = `<strong>Difficulty:</strong> ${escapeHtml(best.difficulty)}<br><strong>Time:</strong> ${escapeHtml(best.time_estimate)}<br><strong>Budget:</strong> ${escapeHtml(best.budget_type || "Unknown")}<br><strong>Format:</strong> ${escapeHtml(best.delivery_mode || "Unknown")}<br><strong>Certificate:</strong> ${best.certificate_available ? "Yes" : "No"}<br><strong>Platform:</strong> ${escapeHtml(best.platform)}`;

    renderChart(recommendations);
    renderCourseCards(recommendations);
    renderCompare(recommendations, data.comparison_table || []);
    renderRefinementPrompts(data.refinement_prompts || []);

    resultsSection.classList.remove("hidden");
    activateTab("dashboard");
    clearStatus();
}

async function fetchJson(path) {
    let lastError = null;

    for (const baseUrl of API_BASE_URLS) {
        try {
            const response = await fetch(`${baseUrl}${path}`);
            if (!response.ok) {
                throw new Error(`Request failed with status ${response.status}`);
            }
            activeApiBaseUrl = baseUrl;
            return await response.json();
        } catch (error) {
            lastError = error;
        }
    }

    throw lastError || new Error("Unable to reach the backend API.");
}

async function fetchMetadata() {
    setStatus("Loading course metadata and filter options...", "success");
    ["level", "learning_style", "delivery_preference", "certificate_preference", "time_commitment", "budget_preference", "preferred_platform"].forEach((id) => {
        setSelectLoadingState(id, "Loading options...");
    });

    const metadata = await fetchJson("/metadata");
    applyMetadata({
        levels: metadata.levels,
        learning_styles: metadata.learning_styles,
        delivery_options: metadata.delivery_options,
        certificate_options: metadata.certificate_options,
        time_commitments: metadata.time_commitments,
        budget_options: metadata.budget_options,
        platforms: metadata.platforms,
    });
    clearStatus();
}

async function fetchRecommendations(payload) {
    const params = new URLSearchParams(payload);
    return fetchJson(`/recommend?${params.toString()}`);
}

async function runRecommendation() {
    const payload = getPayload();
    if (!payload.goal.trim()) {
        setStatus("Describe the student's real goal so the engine can reason about fit instead of popularity.", "warning");
        resultsSection.classList.add("hidden");
        return;
    }

    setFormBusy(true);
    clearStatus();

    try {
        const data = await fetchRecommendations(payload);
        renderResults(data, payload);
    } catch (error) {
        console.error(error);
        setStatus(`Could not fetch recommendations: ${error.message}`, "error");
        resultsSection.classList.add("hidden");
    } finally {
        setFormBusy(false);
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await runRecommendation();
});

limitInput.addEventListener("input", () => {
    limitValue.textContent = limitInput.value;
});

tabButtons.forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.tab));
});

window.addEventListener("DOMContentLoaded", async () => {
    applyMetadata(fallbackMetadata);
    limitInput.value = defaults.limit;
    limitValue.textContent = defaults.limit;

    try {
        await fetchMetadata();
    } catch (error) {
        console.error(error);
        setStatus(`Backend not reachable: ${error.message}. Fallback filter options are loaded; start FastAPI with uvicorn api:app --reload for live recommendations.`, "error");
    }
});
