/**
 * MyLawLLM - Legal Intelligence Interface
 * Core Logic - Production Grade
 */

const App = {
  elements: {
    form: document.getElementById('query-form'),
    input: document.getElementById('user-input'),
    chatHistory: document.getElementById('chat-history'),
    sourcesList: document.getElementById('sources-list'),
    sendBtn: document.getElementById('send-btn')
  },
  
  state: {
    history: [],
    isProcessing: false
  },

  init() {
    this.bindEvents();
    this.elements.input.focus();
    console.log('⚖️ MyLawLLM Intelligence System Active');
  },

  bindEvents() {
    this.elements.form.addEventListener('submit', (e) => {
      e.preventDefault();
      this.handleUserQuery();
    });

    this.elements.input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.handleUserQuery();
      }
    });

    this.elements.input.addEventListener('input', () => {
      this.autoResizeInput();
    });
  },

  autoResizeInput() {
    const input = this.elements.input;
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
  },

  async handleUserQuery(overrideText = null) {
    if (this.state.isProcessing) return;
    
    const query = (overrideText || this.elements.input.value).trim();
    if (!query) return;

    // Reset UI
    if (!overrideText) this.elements.input.value = '';
    this.autoResizeInput();
    this.setProcessing(true);

    // Add User Message
    this.addMessage('user', query);

    // Initial Loading State
    const loadingMessageId = this.addLoadingMessage();
    this.scrollToBottom();

    try {
      const response = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: query,
          history: this.state.history
        })
      });

      if (!response.ok) throw new Error('System unavailable');

      const data = await response.json();
      
      // Update UI with Response
      this.removeMessage(loadingMessageId);
      this.addMessage('bot', data.answer, true);
      this.renderSources(data.sources);

      // Update State
      this.state.history.push({ role: 'user', content: query });
      this.state.history.push({ role: 'assistant', content: data.answer });

    } catch (error) {
      console.error(error);
      this.removeMessage(loadingMessageId);
      this.addMessage('bot', '⚠️ Error: Unable to establish secure connection to legal database. Please try again later.');
    } finally {
      this.setProcessing(false);
      this.scrollToBottom();
    }
  },

  addMessage(role, content, isRich = false) {
    const id = Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.id = `msg-${id}`;

    if (role === 'user') {
      messageDiv.innerHTML = `
        <div class="user-bubble">${this.escapeHtml(content)}</div>
      `;
    } else {
      const formattedContent = isRich ? this.parseLegalResponse(content) : `<p>${content}</p>`;
      messageDiv.innerHTML = `
        <div class="bot-avatar">⚖️</div>
        <div class="message-content">
          ${formattedContent}
        </div>
      `;
    }

    this.elements.chatHistory.appendChild(messageDiv);
    return id;
  },

  addLoadingMessage() {
    const id = 'loading-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot';
    messageDiv.id = id;
    messageDiv.innerHTML = `
      <div class="bot-avatar">⚖️</div>
      <div class="message-content">
        <div class="loading-indicator">
          <div class="loading-dot"></div>
          <div class="loading-dot"></div>
          <div class="loading-dot"></div>
        </div>
      </div>
    `;
    this.elements.chatHistory.appendChild(messageDiv);
    return id;
  },

  removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  },

  parseLegalResponse(text) {
    // Premium Parser for Structured Legal Output
    let html = '';
    
    // Extract sections. Handle both '**' and '##' markdown.
    const plainEnglishRegex = /(?:##|\*\*)*\s*Plain[- ]English Answer(?:[:\*\*]*)\s*([\s\S]*?)(?=(?:##|\*\*)*\s*Legal Basis|$)/i;
    const legalBasisRegex = /(?:##|\*\*)*\s*Legal Basis(?:[:\*\*]*)\s*([\s\S]*)/i;

    const plainMatch = text.match(plainEnglishRegex);
    const legalMatch = text.match(legalBasisRegex);

    if (plainMatch) {
      // Remove any trailing markdown artifacts like '##' that might have been caught before lookahead
      let plainText = plainMatch[1].replace(/(?:##|\*\*)\s*$/, '').trim();
      html += `
        <div class="response-section">
          <span class="section-label">Plain-English Answer</span>
          <div class="section-body">${this.formatListAndParagraphs(plainText)}</div>
        </div>
      `;
    }

    if (legalMatch) {
      html += `
        <div class="response-section" style="border-left: 3px solid var(--accent-gold); margin-top: 16px;">
          <span class="section-label">Legal Basis</span>
          <div class="section-body">${this.formatListAndParagraphs(legalMatch[1].trim())}</div>
        </div>
      `;
    }

    // Fallback if structure isn't perfect
    if (!plainMatch && !legalMatch) {
      html = `<div class="response-section"><div class="section-body">${this.formatListAndParagraphs(text)}</div></div>`;
    }

    return html;
  },

  formatListAndParagraphs(text) {
    // Handle bullet points, numbers, and paragraphs
    let lines = text.split('\n').map(l => l.trim()).filter(l => l);
    let result = '';
    let inList = false;

    lines.forEach(line => {
      // Check for list items: -, *, •, 1.
      if (/^[-\*•\d\.]/.test(line)) {
        if (!inList) {
          result += '<ul class="section-list">';
          inList = true;
        }
        // Clean the prefix
        const cleanItem = line.replace(/^[-\*•\s\d\.]+/, '').trim();
        result += `<li>${this.highlightCitations(cleanItem)}</li>`;
      } else {
        if (inList) {
          result += '</ul>';
          inList = false;
        }
        result += `<p>${this.highlightCitations(line)}</p>`;
      }
    });

    if (inList) result += '</ul>';
    return result || text;
  },

  highlightCitations(text) {
    // Highlight Acts and Sections
    return text.replace(/\b(Section|Sections|S\.)\s*(\d+[A-Z]?)\b/gi, '<span class="citation">$1 $2</span>')
               .replace(/\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Act|Ordinance|Code|Law))\b/g, '<span class="citation">$1</span>');
  },

  renderSources(sources) {
    if (!sources || sources.length === 0) return;
    
    this.elements.sourcesList.innerHTML = '';
    
    sources.forEach(source => {
      const card = document.createElement('div');
      card.className = 'source-card';
      card.innerHTML = `
        <div class="source-name">${this.formatSourceName(source.source)}</div>
        <div class="source-excerpt">${source.excerpt}...</div>
      `;
      card.onclick = () => {
        // Feature: Highlight/Scroll to message that cited this (future enhancement)
        window.alert(`Full reference from: ${this.formatSourceName(source.source)}\n\nExcerpt: ${source.excerpt}`);
      };
      this.elements.sourcesList.appendChild(card);
    });
  },

  formatSourceName(path) {
    return path.replace(/_/g, ' ').replace(/\.pdf$/i, '').replace(/SL\s/i, '');
  },

  setProcessing(isProc) {
    this.state.isProcessing = isProc;
    this.elements.sendBtn.disabled = isProc;
  },

  scrollToBottom() {
    this.elements.chatHistory.scrollTo({
      top: this.elements.chatHistory.scrollHeight,
      behavior: 'smooth'
    });
  },

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  },

  quickQuery(text) {
    this.handleUserQuery(text);
  }
};

// Global hook for suggestion chips
window.quickQuery = (text) => App.quickQuery(text);

// Launch
document.addEventListener('DOMContentLoaded', () => App.init());