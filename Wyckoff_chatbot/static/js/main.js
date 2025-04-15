// $(document).ready(function() {
//     // Add a welcome message
//     addBotMessage("Hello! I'm your RL-enhanced stock trading assistant. You can ask me about stocks or request analysis by typing 'Analyze [STOCK_SYMBOL]'.");
    
//     // Handle send button click
//     $('#send-button').click(sendMessage);
    
//     // Handle enter key press
//     $('#user-input').keypress(function(e) {
//         if (e.which == 13) {
//             sendMessage();
//             return false;
//         }
//     });
    
//     // Function to send message
//     function sendMessage() {
//         const userInput = $('#user-input').val().trim();
        
//         if (userInput === '') return;
        
//         // Add user message to chat
//         addUserMessage(userInput);
        
//         // Clear input field
//         $('#user-input').val('');
        
//         // Add loading indicator
//         const loadingId = addBotMessage('<span class="loading-dots">Processing</span>');
        
//         // Send message to backend
//         $.ajax({
//             url: '/api/chat',
//             type: 'POST',
//             contentType: 'application/json',
//             data: JSON.stringify({ message: userInput }),
//             success: function(response) {
//                 // Remove loading indicator
//                 $(`#${loadingId}`).remove();
                
//                 if (response.type === 'analysis') {
//                     // Show loading for analysis
//                     const analysisLoadingId = addBotMessage(`<span class="loading-dots">Analyzing ${response.stock}</span>`);
                    
//                     // Send request for stock analysis
//                     $.ajax({
//                         url: '/api/analyze',
//                         type: 'POST',
//                         contentType: 'application/json',
//                         data: JSON.stringify({ stock: response.stock }),
//                         success: function(analysisResponse) {
//                             // Remove loading indicator
//                             $(`#${analysisLoadingId}`).remove();
                            
//                             if (analysisResponse.type === 'analysis_result') {
//                                 // Show analysis result
//                                 addBotMessage(analysisResponse.content.replace(/\n/g, '<br>'));
                                
//                                 // Show chart
//                                 showChart(analysisResponse.dates, analysisResponse.chart_data, response.stock);
//                             } else {
//                                 addBotMessage(analysisResponse.content);
//                             }
//                         },
//                         error: function() {
//                             $(`#${analysisLoadingId}`).remove();
//                             addBotMessage("Sorry, I couldn't complete the analysis. Please try again later.");
//                         }
//                     });
//                 } else {
//                     // Show regular text response
//                     addBotMessage(response.content);
//                 }
//             },
//             error: function() {
//                 $(`#${loadingId}`).remove();
//                 addBotMessage("Sorry, I couldn't process your request. Please try again.");
//             }
//         });
//     }
    
//     // Function to add user message to chat
//     function addUserMessage(message) {
//         const messageId = 'msg-' + Date.now();
//         $('#chat-messages').append(`
//             <div id="${messageId}" class="message user-message">
//                 ${message}
//             </div>
//             <div style="clear: both;"></div>
//         `);
//         scrollToBottom();
//         return messageId;
//     }
    
//     // Function to add bot message to chat
//     function addBotMessage(message) {
//         const messageId = 'msg-' + Date.now();
//         $('#chat-messages').append(`
//             <div id="${messageId}" class="message bot-message">
//                 ${message}
//             </div>
//             <div style="clear: both;"></div>
//         `);
//         scrollToBottom();
//         return messageId;
//     }
    
//     // Function to scroll chat to bottom
//     function scrollToBottom() {
//         const chatContainer = document.getElementById('chat-container');
//         chatContainer.scrollTop = chatContainer.scrollHeight;
//     }
    
//     // Function to show stock chart
//     function showChart(dates, prices, stockSymbol) {
//         $('#chart-container').show();
        
//         // If a chart already exists, destroy it
//         if (window.stockChart) {
//             window.stockChart.destroy();
//         }
        
//         const ctx = document.getElementById('stockChart').getContext('2d');
//         window.stockChart = new Chart(ctx, {
//             type: 'line',
//             data: {
//                 labels: dates,
//                 datasets: [{
//                     label: `${stockSymbol} Stock Price`,
//                     data: prices,
//                     backgroundColor: 'rgba(54, 162, 235, 0.2)',
//                     borderColor: 'rgba(54, 162, 235, 1)',
//                     borderWidth: 1,
//                     pointRadius: 0
//                 }]
//             },
//             options: {
//                 responsive: true,
//                 scales: {
//                     x: {
//                         ticks: {
//                             maxTicksLimit: 10
//                         }
//                     },
//                     y: {
//                         beginAtZero: false
//                     }
//                 },
//                 plugins: {
//                     tooltip: {
//                         mode: 'index',
//                         intersect: false
//                     }
//                 }
//             }
//         });
//     }
// });
document.addEventListener('DOMContentLoaded', function() {
    // Global variables
    let priceChart = null;
    let portfolioChart = null;
    let currentSymbol = 'NVDA';
    let backtestHistory = [];
    
    // Navigation tabs functionality
    document.getElementById('dashboard-tab').addEventListener('click', function(e) {
        e.preventDefault();
        showSection('dashboard');
    });
    
    document.getElementById('chatbot-tab').addEventListener('click', function(e) {
        e.preventDefault();
        showSection('chatbot');
    });
    
    document.getElementById('backtest-tab').addEventListener('click', function(e) {
        e.preventDefault();
        showSection('backtest');
    });
    
    // Set default dates for backtest form
    const today = new Date();
    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(today.getFullYear() - 1);
    
    document.getElementById('start-date').valueAsDate = oneYearAgo;
    document.getElementById('end-date').valueAsDate = today;
    
    // Initial load of stock data
    loadStockData(currentSymbol);
    
    // Load stock data button click
    document.getElementById('load-stock-btn').addEventListener('click', function() {
        const symbol = document.getElementById('stock-symbol').value.trim().toUpperCase();
        if (symbol) {
            currentSymbol = symbol;
            loadStockData(symbol);
        }
    });
    
    // Run backtest button click
    document.getElementById('run-backtest-btn').addEventListener('click', function() {
        runBacktest(currentSymbol);
    });
    
    // Backtest form submission
    document.getElementById('backtest-form').addEventListener('submit', function(e) {
        e.preventDefault();
        const symbol = document.getElementById('backtest-symbol').value.trim().toUpperCase();
        runBacktest(symbol);
    });
    
    // Chat functionality
    document.getElementById('send-chat-btn').addEventListener('click', sendChatMessage);
    document.getElementById('chat-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendChatMessage();
        }
    });
    
    // Click event for suggestion buttons
    document.querySelectorAll('.suggestion-btn').forEach(button => {
        button.addEventListener('click', function() {
            document.getElementById('chat-input').value = this.textContent.trim();
            sendChatMessage();
        });
    });
    
    // Recent stocks click event
    document.querySelectorAll('#recent-stocks a').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const symbol = this.getAttribute('data-symbol');
            currentSymbol = symbol;
            document.getElementById('stock-symbol').value = symbol;
            loadStockData(symbol);
        });
    });
    
    // Function to switch between content sections
    function showSection(sectionName) {
        // Hide all sections
        document.querySelectorAll('.content-section').forEach(section => {
            section.classList.add('d-none');
        });
        
        // Show the selected section
        document.getElementById(`${sectionName}-content`).classList.remove('d-none');
        
        // Update active tab
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.getElementById(`${sectionName}-tab`).classList.add('active');
    }
    
    // Function to load stock data
    // Function to load stock data
    function loadStockData(symbol) {
        // Update UI to show loading
        document.getElementById('chart-title').textContent = `${symbol} Stock Price (Loading...)`;
    
        // Fetch stock data from the API
        fetch(`/api/stock_data?symbol=${symbol}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
            // Verify the symbol matches what we requested
                const dataSymbol = data.symbol || symbol;
            
            // Update chart title with the correct symbol
                document.getElementById('chart-title').textContent = `${dataSymbol} Stock Price`;
            
            // Render price chart
                renderPriceChart(data, dataSymbol);
            })
            .catch(error => {
                console.error('Error fetching stock data:', error);
                document.getElementById('chart-title').textContent = `Error loading ${symbol}`;
            });
    }

// Function to render price chart
    function renderPriceChart(data, symbol) {
        const ctx = document.getElementById('price-chart').getContext('2d');
    
        // Destroy existing chart if it exists
        if (priceChart) {
            priceChart.destroy();
        }
    
    // Create new chart
        priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: `${symbol} Price`,
                    data: data.prices,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Price ($)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    }
                }
            }
        });
    }
    
    // Function to run backtest
    function runBacktest(symbol) {
        // Show loading state
        document.getElementById('metrics-content').innerHTML = '<p class="text-center">Running backtest...</p>';
        document.getElementById('signals-content').innerHTML = '<p class="text-center">Running backtest...</p>';
        
        // Get dates from form if available, otherwise use defaults
        let startDate = document.getElementById('start-date').value;
        let endDate = document.getElementById('end-date').value;
        
        if (!startDate || !endDate) {
            startDate = oneYearAgo.toISOString().split('T')[0];
            endDate = today.toISOString().split('T')[0];
        }
        
        // Prepare request data
        const requestData = {
            symbol: symbol,
            start_date: startDate,
            end_date: endDate
        };
        
        // Send backtest request
        fetch('/api/backtest', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(results => {
            // Display backtest results
            displayBacktestResults(results, symbol);
            
            // Add to history
            addToBacktestHistory({
                date: new Date().toLocaleString(),
                symbol: symbol,
                period: `${startDate} to ${endDate}`,
                initialCapital: results.initial_value,
                finalValue: results.final_value,
                roi: results.roi
            });
        })
        .catch(error => {
            console.error('Error running backtest:', error);
            document.getElementById('metrics-content').innerHTML = '<p class="text-center text-danger">Error running backtest</p>';
            document.getElementById('signals-content').innerHTML = '<p class="text-center text-danger">Error running backtest</p>';
        });
    }
    
    // Function to display backtest results
    function displayBacktestResults(results, symbol) {
        // Display metrics
        const roi = results.roi.toFixed(2);
        const roiClass = results.roi >= 0 ? 'positive' : 'negative';
        
        const metricsHtml = `
            <div class="metric-item">
                <span>Initial Capital:</span>
                <span class="metric-value">${results.initial_value.toLocaleString()}</span>
            </div>
            <div class="metric-item">
                <span>Final Value:</span>
                <span class="metric-value">${results.final_value.toLocaleString()}</span>
            </div>
            <div class="metric-item">
                <span>Return on Investment:</span>
                <span class="metric-value ${roiClass}">${roi}%</span>
            </div>
            <div class="metric-item">
                <span>Trading Period:</span>
                <span class="metric-value">${results.dates.length} days</span>
            </div>
        `;
        
        document.getElementById('metrics-content').innerHTML = metricsHtml;
        
        // Display signals (last 5)
        let signalsHtml = '<div class="recent-signals"><h6>Recent Signals:</h6>';
        
        // Get last 5 signals or fewer if not enough
        const numSignals = Math.min(5, results.actions.length);
        const startIdx = results.actions.length - numSignals;
        
        for (let i = startIdx; i < results.actions.length; i++) {
            const action = results.actions[i];
            const date = results.dates[i];
            const price = results.execution_prices[i].toFixed(2);
            
            let signalClass = '';
            if (action === 1) signalClass = 'signal-buy';
            else if (action === 2) signalClass = 'signal-sell';
            else signalClass = 'signal-hold';
            
            signalsHtml += `
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span>${date}</span>
                    <span class="signal-badge ${signalClass}">${results.action_labels[action]} @ ${price}</span>
                </div>
            `;
        }
        
        signalsHtml += '</div>';
        
        // Add summary of all signals
        const buys = results.actions.filter(a => a === 1).length;
        const sells = results.actions.filter(a => a === 2).length;
        const holds = results.actions.filter(a => a === 0).length;
        
        signalsHtml += `
            <hr>
            <div class="signal-summary">
                <h6>Signal Summary:</h6>
                <div class="d-flex justify-content-between">
                    <span class="signal-badge signal-buy">Buy: ${buys}</span>
                    <span class="signal-badge signal-sell">Sell: ${sells}</span>
                    <span class="signal-badge signal-hold">Hold: ${holds}</span>
                </div>
            </div>
        `;
        
        document.getElementById('signals-content').innerHTML = signalsHtml;
        
        // Update chart with portfolio values
        updatePriceChartWithBacktest(results);
    }
    
    // Function to update price chart with backtest results
    function updatePriceChartWithBacktest(results) {
        if (!priceChart) return;
        
        // Add portfolio value dataset
        const portfolioDataset = {
            label: 'Portfolio Value',
            data: results.portfolio_values,
            borderColor: 'rgb(153, 102, 255)',
            backgroundColor: 'rgba(153, 102, 255, 0.5)',
            yAxisID: 'y1',
            tension: 0.1
        };
        
        // Add buy/sell markers
        const buyPoints = [];
        const sellPoints = [];
        
        results.actions.forEach((action, i) => {
            if (action === 1) { // Buy
                buyPoints.push({
                    x: results.dates[i],
                    y: results.execution_prices[i]
                });
            } else if (action === 2) { // Sell
                sellPoints.push({
                    x: results.dates[i],
                    y: results.execution_prices[i]
                });
            }
        });
        
        const buyDataset = {
            label: 'Buy Signals',
            data: buyPoints,
            backgroundColor: 'rgba(40, 167, 69, 1)',
            borderColor: 'rgba(40, 167, 69, 1)',
            pointRadius: 5,
            pointStyle: 'triangle',
            showLine: false
        };
        
        const sellDataset = {
            label: 'Sell Signals',
            data: sellPoints,
            backgroundColor: 'rgba(220, 53, 69, 1)',
            borderColor: 'rgba(220, 53, 69, 1)',
            pointRadius: 5,
            pointStyle: 'triangle',
            rotation: 180,
            showLine: false
        };
        
        // Update chart datasets
        priceChart.data.datasets = [
            priceChart.data.datasets[0], // Keep original price dataset
            buyDataset,
            sellDataset
        ];
        
        // Update chart options to include second y-axis for portfolio value
        priceChart.options.scales.y1 = {
            type: 'linear',
            display: false,
            position: 'right'
        };
        
        priceChart.update();
    }
    
    // Function to add backtest to history
    function addToBacktestHistory(backtest) {
        backtestHistory.unshift(backtest); // Add to start of array
        
        // Keep only the last 10 backtests
        if (backtestHistory.length > 10) {
            backtestHistory = backtestHistory.slice(0, 10);
        }
        
        // Update table
        updateBacktestHistoryTable();
    }
    
    // Function to update backtest history table
    function updateBacktestHistoryTable() {
        const tableBody = document.getElementById('backtest-history');
        tableBody.innerHTML = '';
        
        backtestHistory.forEach(backtest => {
            const roi = ((backtest.finalValue / backtest.initialCapital - 1) * 100).toFixed(2);
            const roiClass = roi >= 0 ? 'text-success' : 'text-danger';
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${backtest.date}</td>
                <td>${backtest.symbol}</td>
                <td>${backtest.period}</td>
                <td>${backtest.initialCapital.toLocaleString()}</td>
                <td>${backtest.finalValue.toLocaleString()}</td>
                <td class="${roiClass}">${roi}%</td>
                <td>
                    <button class="btn btn-sm btn-outline-primary load-backtest-btn" 
                            data-symbol="${backtest.symbol}">
                        Load
                    </button>
                </td>
            `;
            
            tableBody.appendChild(row);
        });
        
        // Add event listeners to load buttons
        document.querySelectorAll('.load-backtest-btn').forEach(button => {
            button.addEventListener('click', function() {
                const symbol = this.getAttribute('data-symbol');
                document.getElementById('stock-symbol').value = symbol;
                currentSymbol = symbol;
                loadStockData(symbol);
                showSection('dashboard');
            });
        });
    }
    
    // Function to send chat message
    function sendChatMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message) return;
        
        // Add user message to chat
        addMessageToChat(message, 'user');
        
        // Clear input
        input.value = '';
        
        // Show loading indicator
        const loadingId = addMessageToChat('Thinking...', 'system');
        
        // Send message to API
        fetch('/api/wyckoff_chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading message
            document.getElementById(loadingId).remove();
            
            // Add response to chat
            addMessageToChat(data.response, 'system');
        })
        .catch(error => {
            console.error('Error sending message:', error);
            
            // Remove loading message
            document.getElementById(loadingId).remove();
            
            // Add error message
            addMessageToChat('Sorry, I encountered an error processing your request.', 'system');
        });
    }
    
    // Function to add message to chat
    function addMessageToChat(message, sender) {
        const chatContainer = document.getElementById('chat-messages');
        const messageId = 'msg-' + Date.now();
        
        const messageDiv = document.createElement('div');
        messageDiv.id = messageId;
        messageDiv.className = `message ${sender}-message`;
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <p>${message}</p>
            </div>
        `;
        
        chatContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
        return messageId;
    }
});