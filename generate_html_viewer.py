#!/usr/bin/env python3
"""
Generate a standalone HTML file for interactive viewing of leaderboard CSV results.
"""


def generate_html_viewer(output_path="data/results/leaderboard_viewer.html"):
    """Generate a standalone HTML file for interactive CSV viewing."""

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Leaderboard CSV Viewer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.2em;
        }
        
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        
        .controls {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        
        .file-input-area {
            border: 2px dashed #ddd;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }
        
        .file-input-area:hover,
        .file-input-area.dragover {
            border-color: #3498db;
            background-color: #f8f9fa;
        }
        
        .file-input-area input[type="file"] {
            display: none;
        }
        
        .file-input-label {
            cursor: pointer;
            color: #666;
            font-size: 1.1em;
        }
        
        .file-input-label:hover {
            color: #3498db;
        }
        
        .search-container {
            margin-top: 15px;
        }
        
        .search-input {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        
        .search-input:focus {
            outline: none;
            border-color: #3498db;
        }
        
        .table-container {
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .table-info {
            padding: 15px 25px;
            background-color: #f8f9fa;
            border-bottom: 1px solid #eee;
            font-size: 0.9em;
            color: #666;
        }
        
        .table-wrapper {
            overflow-x: auto;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        
        th {
            background-color: #34495e;
            color: white;
            padding: 15px 10px;
            text-align: left;
            cursor: pointer;
            user-select: none;
            position: sticky;
            top: 0;
            z-index: 10;
            transition: background-color 0.2s ease;
        }
        
        th:hover {
            background-color: #2c3e50;
        }
        
        th.sortable:after {
            content: "⇅";
            margin-left: 8px;
            opacity: 0.5;
        }
        
        th.sort-asc:after {
            content: "▲";
            opacity: 1;
        }
        
        th.sort-desc:after {
            content: "▼";
            opacity: 1;
        }
        
        td {
            padding: 12px 10px;
            border-bottom: 1px solid #eee;
        }
        
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        tr:hover {
            background-color: #e8f4f8;
        }
        
        .highlight-column {
            background-color: #fff3cd;
            font-weight: 600;
        }
        
        .numeric {
            text-align: right;
            font-family: "SF Mono", Monaco, "Cascadia Code", "Roboto Mono", Consolas, "Courier New", monospace;
        }
        
        .no-data {
            text-align: center;
            padding: 60px 20px;
            color: #666;
            font-size: 1.1em;
        }
        
        .loading {
            text-align: center;
            padding: 40px 20px;
            color: #666;
        }
        
        .error {
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        .ned-value {
            color: #666;
            font-style: italic;
            position: relative;
            cursor: help;
        }
        
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        .tooltip .tooltip-text {
            visibility: hidden;
            width: 120px;
            background-color: #2c3e50;
            color: white;
            text-align: center;
            border-radius: 6px;
            padding: 8px 12px;
            position: absolute;
            z-index: 999;
            bottom: 125%;
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 12px;
            font-style: normal;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        
        .tooltip .tooltip-text::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #2c3e50 transparent transparent transparent;
        }
        
        .tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header, .controls {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 1.8em;
            }
            
            table {
                font-size: 12px;
            }
            
            th, td {
                padding: 8px 6px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏆 Leaderboard CSV Viewer</h1>
            <p>Interactive viewer for ForecastBench leaderboard results with sorting and filtering capabilities</p>
        </div>
        
        <div class="controls">
            <div class="file-input-area" id="fileInputArea">
                <label for="csvFile" class="file-input-label">
                    📁 Click to select a CSV file or drag & drop here
                </label>
                <input type="file" id="csvFile" accept=".csv" />
            </div>
            
            <div class="search-container">
                <input 
                    type="text" 
                    id="searchInput" 
                    class="search-input" 
                    placeholder="🔍 Search by model name..." 
                    disabled
                />
            </div>
        </div>
        
        <div class="table-container">
            <div class="table-info" id="tableInfo" style="display: none;">
                Showing <span id="rowCount">0</span> models
            </div>
            
            <div class="table-wrapper">
                <div class="no-data" id="noData">
                    Load a CSV file to view the leaderboard data
                </div>
                
                <table id="dataTable" style="display: none;">
                    <thead id="tableHead"></thead>
                    <tbody id="tableBody"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        class LeaderboardViewer {
            constructor() {
                this.data = [];
                this.filteredData = [];
                this.sortColumn = null;
                this.sortDirection = 'asc';
                this.mainScoreColumn = 'diff_adj_brier_score';
                
                this.initializeElements();
                this.setupEventListeners();
            }
            
            getCleanColumnName(columnName) {
                const columnMapping = {
                    'model': 'Model',
                    'organization': 'Organization',
                    'n_forecasts_market': 'N Forecasts (Market)',
                    'n_forecasts_dataset': 'N Forecasts (Dataset)',
                    'model_days_active_market': 'Days Active (Market)',
                    'model_days_active_dataset': 'Days Active (Dataset)',
                    'model_n_forecasts_market': 'N Forecasts (Market)',
                    'model_n_forecasts_dataset': 'N Forecasts (Dataset)',
                    'diff_adj_brier_score': 'Score',
                    'diff_adj_brier_score_market': 'Score (Market)',
                    'diff_adj_brier_score_dataset': 'Score (Dataset)',
                    'diff_adj_log_score': 'Score',
                    'diff_adj_log_score_market': 'Score (Market)', 
                    'diff_adj_log_score_dataset': 'Score (Dataset)'
                };
                return columnMapping[columnName] || columnName;
            }
            
            isIntegerColumn(columnName) {
                return columnName.includes('days_active') || columnName.includes('n_forecasts');
            }
            
            initializeElements() {
                this.fileInput = document.getElementById('csvFile');
                this.fileInputArea = document.getElementById('fileInputArea');
                this.searchInput = document.getElementById('searchInput');
                this.tableInfo = document.getElementById('tableInfo');
                this.rowCount = document.getElementById('rowCount');
                this.noData = document.getElementById('noData');
                this.dataTable = document.getElementById('dataTable');
                this.tableHead = document.getElementById('tableHead');
                this.tableBody = document.getElementById('tableBody');
            }
            
            setupEventListeners() {
                // File input
                this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
                
                // Drag and drop
                this.fileInputArea.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    this.fileInputArea.classList.add('dragover');
                });
                
                this.fileInputArea.addEventListener('dragleave', () => {
                    this.fileInputArea.classList.remove('dragover');
                });
                
                this.fileInputArea.addEventListener('drop', (e) => {
                    e.preventDefault();
                    this.fileInputArea.classList.remove('dragover');
                    const files = e.dataTransfer.files;
                    if (files.length > 0) {
                        this.handleFile(files[0]);
                    }
                });
                
                // Search input
                this.searchInput.addEventListener('input', () => this.handleSearch());
            }
            
            handleFileSelect(event) {
                const file = event.target.files[0];
                if (file) {
                    this.handleFile(file);
                }
            }
            
            handleFile(file) {
                if (!file.name.toLowerCase().endsWith('.csv')) {
                    this.showError('Please select a CSV file');
                    return;
                }
                
                this.showLoading();
                
                const reader = new FileReader();
                reader.onload = (e) => {
                    try {
                        this.parseCSV(e.target.result, file.name);
                    } catch (error) {
                        this.showError('Error parsing CSV: ' + error.message);
                    }
                };
                reader.readAsText(file);
            }
            
            parseCSV(csvText, filename) {
                const lines = csvText.trim().split('\\n');
                if (lines.length < 2) {
                    throw new Error('CSV file must have at least a header and one data row');
                }
                
                const headers = this.parseCSVLine(lines[0]);
                const data = [];
                
                for (let i = 1; i < lines.length; i++) {
                    const values = this.parseCSVLine(lines[i]);
                    if (values.length === headers.length) {
                        const row = {};
                        headers.forEach((header, index) => {
                            row[header] = values[index];
                        });
                        data.push(row);
                    }
                }
                
                this.data = data;
                this.filteredData = [...data];
                this.createTable(headers, filename);
                this.enableSearch();
            }
            
            parseCSVLine(line) {
                const result = [];
                let current = '';
                let inQuotes = false;
                
                for (let i = 0; i < line.length; i++) {
                    const char = line[i];
                    
                    if (char === '"') {
                        inQuotes = !inQuotes;
                    } else if (char === ',' && !inQuotes) {
                        result.push(current.trim());
                        current = '';
                    } else {
                        current += char;
                    }
                }
                result.push(current.trim());
                
                return result;
            }
            
            createTable(headers, filename) {
                // Reorder headers to put 'organization' right after 'model'
                const reorderedHeaders = [];
                let modelIndex = -1;
                let orgIndex = -1;
                
                // Find the indices
                headers.forEach((header, index) => {
                    if (header === 'model') modelIndex = index;
                    if (header === 'organization') orgIndex = index;
                });
                
                // Build reordered array
                headers.forEach((header, index) => {
                    if (header === 'model') {
                        reorderedHeaders.push(header);
                        // Add organization right after model if it exists
                        if (orgIndex !== -1) {
                            reorderedHeaders.push('organization');
                        }
                    } else if (header !== 'organization') {
                        reorderedHeaders.push(header);
                    }
                });
                
                // Create header
                const headerRow = document.createElement('tr');
                reorderedHeaders.forEach(header => {
                    const th = document.createElement('th');
                    th.textContent = this.getCleanColumnName(header);
                    th.classList.add('sortable');
                    th.dataset.column = header;
                    
                    th.addEventListener('click', () => this.sortTable(header));
                    headerRow.appendChild(th);
                });
                
                // Store the reordered headers for use in renderTableBody
                this.columnOrder = reorderedHeaders;
                
                this.tableHead.innerHTML = '';
                this.tableHead.appendChild(headerRow);
                
                this.renderTableBody();
                this.updateTableInfo(filename);
                this.showTable();
            }
            
            renderTableBody() {
                this.tableBody.innerHTML = '';
                
                this.filteredData.forEach(row => {
                    const tr = document.createElement('tr');
                    
                    // Use the reordered column order if available, otherwise use original keys
                    const keysToIterate = this.columnOrder || Object.keys(row);
                    
                    keysToIterate.forEach(key => {
                        const td = document.createElement('td');
                        const value = row[key];
                        
                        // Check if value is NULL/empty/undefined
                        const isNull = value === '' || value === null || value === undefined || value === 'NULL';
                        
                        if (isNull) {
                            // Create tooltip wrapper for n.e.d.
                            const tooltipSpan = document.createElement('span');
                            tooltipSpan.className = 'tooltip ned-value';
                            tooltipSpan.textContent = 'n.e.d.';
                            
                            const tooltipText = document.createElement('span');
                            tooltipText.className = 'tooltip-text';
                            tooltipText.textContent = 'not enough data';
                            
                            tooltipSpan.appendChild(tooltipText);
                            td.appendChild(tooltipSpan);
                        } else if (this.isNumeric(value)) {
                            // Format numeric columns
                            td.classList.add('numeric');
                            const num = parseFloat(value);
                            if (this.isIntegerColumn(key)) {
                                td.textContent = Math.round(num).toString();
                            } else if (this.isScoreColumn(key)) {
                                td.textContent = this.formatToSignificantDigits(num, 3);
                            } else {
                                td.textContent = num.toFixed(4);
                            }
                        } else {
                            td.textContent = value;
                        }
                        
                        tr.appendChild(td);
                    });
                    
                    this.tableBody.appendChild(tr);
                });
                
                this.updateRowCount();
            }
            
            isNumeric(value) {
                return !isNaN(parseFloat(value)) && isFinite(value);
            }
            
            formatToSignificantDigits(num, digits) {
                if (num === 0) return '0';
                const magnitude = Math.floor(Math.log10(Math.abs(num)));
                const factor = Math.pow(10, digits - 1 - magnitude);
                return (Math.round(num * factor) / factor).toString();
            }
            
            isScoreColumn(columnName) {
                return columnName.includes('diff_adj_brier_score');
            }
            
            sortTable(column) {
                // Update sort direction
                if (this.sortColumn === column) {
                    this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
                } else {
                    this.sortColumn = column;
                    this.sortDirection = 'asc';
                }
                
                // Update header indicators
                this.updateSortIndicators(column);
                
                // Sort the data
                this.filteredData.sort((a, b) => {
                    let aVal = a[column];
                    let bVal = b[column];
                    
                    // Handle NULL/empty values - always put them at the bottom
                    const aIsNull = aVal === '' || aVal === null || aVal === undefined || aVal === 'NULL';
                    const bIsNull = bVal === '' || bVal === null || bVal === undefined || bVal === 'NULL';
                    
                    if (aIsNull && bIsNull) return 0;
                    if (aIsNull) return 1;  // a goes to bottom
                    if (bIsNull) return -1; // b goes to bottom
                    
                    // Handle numeric values
                    if (this.isNumeric(aVal) && this.isNumeric(bVal)) {
                        aVal = parseFloat(aVal);
                        bVal = parseFloat(bVal);
                    }
                    
                    let comparison = 0;
                    if (aVal < bVal) comparison = -1;
                    else if (aVal > bVal) comparison = 1;
                    
                    return this.sortDirection === 'asc' ? comparison : -comparison;
                });
                
                this.renderTableBody();
            }
            
            updateSortIndicators(activeColumn) {
                const headers = this.tableHead.querySelectorAll('th');
                headers.forEach(th => {
                    th.classList.remove('sort-asc', 'sort-desc');
                    if (th.dataset.column === activeColumn) {
                        th.classList.add(`sort-${this.sortDirection}`);
                    }
                });
            }
            
            handleSearch() {
                const searchTerm = this.searchInput.value.toLowerCase();
                
                if (!searchTerm) {
                    this.filteredData = [...this.data];
                } else {
                    this.filteredData = this.data.filter(row => {
                        return row.model && row.model.toLowerCase().includes(searchTerm);
                    });
                }
                
                this.renderTableBody();
            }
            
            enableSearch() {
                this.searchInput.disabled = false;
            }
            
            updateTableInfo(filename) {
                this.tableInfo.style.display = 'block';
                this.tableInfo.querySelector('#rowCount').textContent = this.data.length;
            }
            
            updateRowCount() {
                this.rowCount.textContent = this.filteredData.length;
            }
            
            showTable() {
                this.noData.style.display = 'none';
                this.dataTable.style.display = 'table';
            }
            
            showLoading() {
                this.noData.innerHTML = '<div class="loading">Loading CSV file...</div>';
            }
            
            showError(message) {
                this.noData.innerHTML = `<div class="error">${message}</div>`;
            }
        }
        
        // Initialize the viewer when the page loads
        document.addEventListener('DOMContentLoaded', () => {
            new LeaderboardViewer();
        });
    </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


if __name__ == "__main__":
    output_file = generate_html_viewer()
    print(f"HTML viewer generated: {output_file}")
