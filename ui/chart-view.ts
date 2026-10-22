/// <reference lib="dom" />
import { Chart, Legend, LinearScale, LineController, LineElement, PointElement, Tooltip } from "chart.js";
import type { ChartPoint } from "./series.ts";

Chart.register(LineController, LineElement, PointElement, LinearScale, Legend, Tooltip);

export class ChartPanel {
  private readonly errorChart: Chart;
  private readonly sigmaChart: Chart;

  constructor(errorCanvas: HTMLCanvasElement, sigmaCanvas: HTMLCanvasElement) {
    this.errorChart = lineChart(errorCanvas, "Position error", "#c2410c");
    this.sigmaChart = lineChart(sigmaCanvas, "Uncertainty (2σ)", "#1e3a5f");
  }

  setSeries(error: readonly ChartPoint[], sigma: readonly ChartPoint[]): void {
    setPoints(this.errorChart, error);
    setPoints(this.sigmaChart, sigma);
  }
}

function lineChart(canvas: HTMLCanvasElement, label: string, color: string): Chart {
  return new Chart(canvas, {
    type: "line",
    data: {
      datasets: [
        {
          label,
          data: [],
          borderColor: color,
          backgroundColor: color,
          pointRadius: 0,
          borderWidth: 2,
          tension: 0.15,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: true, labels: { boxWidth: 12 } } },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Time (s)" },
          ticks: { maxTicksLimit: 8 },
        },
        y: {
          type: "linear",
          beginAtZero: true,
          title: { display: true, text: "Meters" },
        },
      },
    },
  });
}

function setPoints(chart: Chart, points: readonly ChartPoint[]): void {
  const dataset = chart.data.datasets[0];
  dataset.data = points.map((point) => ({ x: point.t, y: point.value }));
  chart.update();
}
