// Run History tab: list past runs, click through to detail + jump into
// the Watch tab for post-hoc replay.
const RunHistory = {
  async init() {
    await this.refresh();
  },

  async refresh() {
    const runs = await Api.listRuns(50);
    const tbody = document.querySelector("#runsTable tbody");
    tbody.innerHTML = "";
    for (const r of runs) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${r.run_id}</td><td>${r.layout_id ?? ""}</td><td>${r.status}</td>
        <td>${r.seed}</td><td>${r.started_at ? new Date(r.started_at).toLocaleString() : ""}</td>
        <td>${r.final_tick ?? ""}</td>
        <td><button data-action="watch" data-run="${r.run_id}">Watch/Replay</button>
            <button data-action="report" data-run="${r.run_id}">Report</button></td>`;
      tr.querySelector("td:first-child").addEventListener("click", () => this.showDetail(r.run_id));
      tr.querySelector('button[data-action="watch"]').addEventListener("click", (e) => {
        e.stopPropagation();
        Main.goToTab("watch");
        document.getElementById("watchRunId").value = r.run_id;
        SimView.watch(r.run_id);
      });
      tr.querySelector('button[data-action="report"]').addEventListener("click", (e) => {
        e.stopPropagation();
        Main.goToTab("report");
        document.getElementById("reportRunId").value = r.run_id;
        ReportView.load(r.run_id);
      });
      tbody.appendChild(tr);
    }
  },

  async showDetail(runId) {
    const detail = await Api.getRun(runId);
    document.getElementById("runDetail").textContent = JSON.stringify(detail, null, 2);
  },
};
