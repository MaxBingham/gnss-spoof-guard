import type { Cov2, ImuSample } from "../sensors/types.ts";
import { wrapAngle } from "./angles.ts";
import type { Estimate } from "./types.ts";

/**
 * Horizontal 6-state Kalman filter: x, y, vx, vy, accelerometer bias x, bias y.
 * Altitude is a separate 2-state barometer filter. Heading integrates gyro rate
 * and corrects with the compass. Position fixes correct the horizontal filter.
 */
export class KalmanFilter {
  private x = [0, 0, 0, 0, 0, 0];
  private P = diagonal([25, 25, 1, 1, 0.05, 0.05]);
  private alt = [0, 0];
  private Pa = diagonal([4, 1]);
  private yaw = 0;

  predict(imu: ImuSample, dt: number): void {
    const dt2 = 0.5 * dt * dt;
    const F = identity(6);
    F[0][2] = dt;
    F[0][4] = -dt2;
    F[1][3] = dt;
    F[1][5] = -dt2;
    F[2][4] = -dt;
    F[3][5] = -dt;
    const Q = diagonal([0.02 * dt, 0.02 * dt, 0.3 * dt, 0.3 * dt, 1e-5 * dt, 1e-5 * dt]);
    this.P = symmetrize(add(mul(mul(F, this.P), transpose(F)), Q));

    const ax = imu.ax - this.x[4];
    const ay = imu.ay - this.x[5];
    this.x[0] += this.x[2] * dt + ax * dt2;
    this.x[1] += this.x[3] * dt + ay * dt2;
    this.x[2] += ax * dt;
    this.x[3] += ay * dt;

    this.predictAltitude(imu.az, dt);
    this.yaw = wrapAngle(this.yaw + imu.yawRateRadS * dt);
  }

  updatePosition(x: number, y: number, cov: Cov2): void {
    const H = [
      [1, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0],
    ];
    const R = [
      [cov[0], cov[1]],
      [cov[1], cov[2]],
    ];
    const PHt = mul(this.P, transpose(H));
    const S = add(mul(H, PHt), R);
    const K = mul(PHt, inv2(S));
    const innovationX = x - this.x[0];
    const innovationY = y - this.x[1];
    for (let i = 0; i < 6; i++) this.x[i] += K[i][0] * innovationX + K[i][1] * innovationY;

    const IKH = sub(identity(6), mul(K, H));
    const KRKt = mul(mul(K, R), transpose(K));
    this.P = symmetrize(add(mul(mul(IKH, this.P), transpose(IKH)), KRKt));
  }

  updateBaro(altitudeM: number): void {
    const R = 0.25;
    const s = this.Pa[0][0] + R;
    const kz = this.Pa[0][0] / s;
    const kv = this.Pa[1][0] / s;
    const innovation = altitudeM - this.alt[0];
    this.alt[0] += kz * innovation;
    this.alt[1] += kv * innovation;
    const p00 = (1 - kz) * this.Pa[0][0];
    const p01 = (1 - kz) * this.Pa[0][1];
    const p10 = this.Pa[1][0] - kv * this.Pa[0][0];
    const p11 = this.Pa[1][1] - kv * this.Pa[0][1];
    this.Pa = symmetrize([
      [p00, p01],
      [p10, p11],
    ]);
  }

  updateCompass(yawRad: number): void {
    this.yaw = wrapAngle(this.yaw + 0.15 * wrapAngle(yawRad - this.yaw));
  }

  estimate(): Estimate {
    return {
      x: this.x[0],
      y: this.x[1],
      vx: this.x[2],
      vy: this.x[3],
      z: this.alt[0],
      yaw: this.yaw,
      cov: [this.P[0][0], this.P[0][1], this.P[1][1]],
    };
  }

  private predictAltitude(az: number, dt: number): void {
    const dt2 = 0.5 * dt * dt;
    this.alt[0] += this.alt[1] * dt + az * dt2;
    this.alt[1] += az * dt;
    const F = [
      [1, dt],
      [0, 1],
    ];
    const Q = diagonal([0.05 * dt, 0.2 * dt]);
    this.Pa = symmetrize(add(mul(mul(F, this.Pa), transpose(F)), Q));
  }
}

function diagonal(values: number[]): number[][] {
  return values.map((value, row) => values.map((_, col) => (row === col ? value : 0)));
}

function identity(n: number): number[][] {
  return diagonal(Array.from({ length: n }, () => 1));
}

function zeros(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0));
}

function mul(a: number[][], b: number[][]): number[][] {
  const out = zeros(a.length, b[0].length);
  for (let i = 0; i < a.length; i++) {
    for (let k = 0; k < b.length; k++) {
      const aik = a[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < b[0].length; j++) out[i][j] += aik * b[k][j];
    }
  }
  return out;
}

function add(a: number[][], b: number[][]): number[][] {
  return a.map((row, i) => row.map((value, j) => value + b[i][j]));
}

function sub(a: number[][], b: number[][]): number[][] {
  return a.map((row, i) => row.map((value, j) => value - b[i][j]));
}

function transpose(a: number[][]): number[][] {
  const out = zeros(a[0].length, a.length);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[0].length; j++) out[j][i] = a[i][j];
  }
  return out;
}

function inv2(m: number[][]): number[][] {
  const a = m[0][0];
  const b = m[0][1];
  const c = m[1][0];
  const d = m[1][1];
  const det = a * d - b * c;
  const scale = 1 / det;
  return [
    [d * scale, -b * scale],
    [-c * scale, a * scale],
  ];
}

function symmetrize(m: number[][]): number[][] {
  return m.map((row, i) => row.map((value, j) => 0.5 * (value + m[j][i])));
}
