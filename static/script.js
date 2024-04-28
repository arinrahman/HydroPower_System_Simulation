$(document).ready(function () {
    $('#dark-mode-toggle').click(function () {
        $.post('/toggle-dark-mode', function (data) {
            $('body').toggleClass('dark-mode');
        });
    });
});


$(document).ready(function () {
    function updateRangeInputValue(inputId) {
        var value = $('#' + inputId).val();
        $('#' + inputId + '-value').text(value);
    }

    // Update range input values when user interacts with them
    $('#temperature, #release, #inflow').on('input', function () {
        updateRangeInputValue($(this).attr('id'));
    });
    
    $('#controls-form input[type=range]').change(function () {
        submitForm();
    });

    submitForm();
});

$(document).ready(function () {
    var years = 0;
    var days = 0;
    var hours = 0;

    function updateTime() {
        hours++;
        if (hours >= 24) {
            days++;
            hours = 0;
        }
        if (days >= 365) {
            years++;
            days = 0;
        }

        $('.time #time-value').text(years + ' years, ' + days + ' days, ' + hours + ' hours');

        submitForm();
    }

    function updateEnergyOutput() {
        $.ajax({
            type: 'GET',
            url: '/get_energy_output',
            success: function (data) {
                var formattedOutput = parseFloat(data.energy_output).toFixed(2);
            
                $('.energy-output #energy-output-value').text(formattedOutput);
                updateChart(); 
            },
            error: function (xhr, status, error) {
                console.error(error);
            }
        });
    }
    setInterval(function () {
        updateEnergyOutput();
        updateTime();
    }, 1000);
});

function submitForm() {
    $.ajax({
        type: 'POST',
        url: '/update',
        data: $('#controls-form').serialize(),
        success: function (data) {
            var waterLevel = data.water_level;
            $('#water').css('height', waterLevel + '%');
        },
        error: function (xhr, status, error) {
            console.error(error);
        }
    });
}

function updateChart() {
    $.ajax({
        type: 'GET',
        url: '/get_water_level_data',
        success: function (data) {
            var labels = [];
            var values = [];

            data.forEach(function (entry) {
                labels.push(entry.time);
                values.push(entry.water_level);
            });

            myChart.data.labels = labels;
            myChart.data.datasets[0].data = values;
            myChart.update();
        },
        error: function (xhr, status, error) {
            console.error(error);
        }
    });
}

