// import { Component, signal } from '@angular/core';
// import { RouterOutlet } from '@angular/router';

// @Component({
//   selector: 'app-root',
//   imports: [RouterOutlet],
//   templateUrl: './app.html',
//   styleUrl: './app.css'
// })
// export class App {
//   protected readonly title = signal('angular');
// }

// src/app/app.ts
import { Component } from '@angular/core';
import { NgIf, JsonPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [NgIf, JsonPipe, FormsModule, HttpClientModule],
  templateUrl: './app.html',
  styleUrls: ['./app.css'],
})
export class App {
  title = 'SLM Routing UI';
  text = '';
  result: any = null;
  loading = false;

  // Read from assets/env.js (added below)
  backendUrl = (window as any).env?.BACKEND_URL || 'http://127.0.0.1:3001';

  constructor(private http: HttpClient) {}

  route() {
    const q = this.text.trim();
    if (!q) return;
    this.loading = true;
    this.http.post<any>(`${this.backendUrl}/route`, { text: q }).subscribe({
      next: (r) => { this.result = r; this.loading = false; },
      error: () => { this.result = { error: 'routing_failed' }; this.loading = false; },
    });
  }
}

