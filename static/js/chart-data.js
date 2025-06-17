window.onload = function() {
    // PIE CHART
    new Chart(document.getElementById('pieChart'), {
        type: 'pie',
        data: {
            labels: ['Falling', 'Punching', 'Kicking'],
            datasets: [{
                data: [6, 10, 6],
                backgroundColor: ['green', 'yellow', 'red']
            }]
        }
    });

    // BAR CHART (hourly)
    new Chart(document.getElementById('barChart'), {
        type: 'bar',
        data: {
            labels: Array.from({length: 24}, (_, i) => `${i}:00`),
            datasets: [{
                label: 'Hazardous Behaviors',
                data: Array.from({length: 24}, () => Math.floor(Math.random() * 3)),
                backgroundColor: 'blue'
            }]
        }
    });

    // LINE CHART
    new Chart(document.getElementById('lineChart'), {
        type: 'line',
        data: {
            labels: Array.from({length: 30}, (_, i) => `Day ${i + 1}`),
            datasets: [{
                label: 'Hazardous Behaviors',
                data: Array.from({length: 30}, () => Math.floor(Math.random() * 3)),
                borderColor: 'green',
                fill: false
            }]
        }
    });

    // TODAY BAR
    new Chart(document.getElementById('barToday'), {
        type: 'bar',
        data: {
            labels: ['Falling', 'Punching', 'Kicking'],
            datasets: [{
                label: 'Count',
                data: [1, 14, 9],
                backgroundColor: ['green', 'yellow', 'red']
            }]
        }
    });

    // MONTHLY BAR
    new Chart(document.getElementById('barMonth'), {
        type: 'bar',
        data: {
            labels: ['Falling', 'Punching', 'Kicking'],
            datasets: [{
                label: 'Count',
                data: [2, 13, 10],
                backgroundColor: ['green', 'yellow', 'red']
            }]
        }
    });
};
