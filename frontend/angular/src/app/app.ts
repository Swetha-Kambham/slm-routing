import { Component, OnInit } from '@angular/core';
import { NgIf, NgFor, JsonPipe, DecimalPipe } from '@angular/common'; // 👈 add DecimalPipe
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [NgIf, NgFor, JsonPipe, DecimalPipe, FormsModule, HttpClientModule], // 👈 add here
  templateUrl: './app.html',
  styleUrls: ['./app.css'],
})
export class App implements OnInit {
  title = 'SLM Routing UI';
  text = '';
  result: any = null;
  loading = false;

  backendUrl = (window as any).env?.BACKEND_URL || 'http://127.0.0.1:3001';
  config: { model_url: string; threshold: number; labels: string[] } | null = null;
  backendHealthy = false;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    // fetch config (labels, threshold, model url)
    this.http.get<any>(`${this.backendUrl}/config`).subscribe({
      next: (c) => { this.config = c; this.backendHealthy = true; },
      error: () => { this.config = null; this.backendHealthy = false; },
    });
  }

  route() {
    const q = this.text.trim();
    if (!q) return;
    this.loading = true;
    this.http.post<any>(`${this.backendUrl}/route`, { text: q }).subscribe({
      next: (r) => { this.result = r; this.loading = false; },
      error: () => { this.result = { error: 'routing_failed' }; this.loading = false; },
    });
  }

  ping() {
    this.http.get<any>(`${this.backendUrl}/healthz`).subscribe({
      next: () => this.backendHealthy = true,
      error: () => this.backendHealthy = false,
    });
  }
}
