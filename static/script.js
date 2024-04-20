$(document).ready(function () {
    $('#dark-mode-toggle').click(function () {
        $.post('/toggle-dark-mode', function (data) {
            $('body').toggleClass('dark-mode');
        });
    });
});


$(document).ready(function () {

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
        // Ajax request to fetch energy output from the server
        $.ajax({
            type: 'GET',
            url: '/get_energy_output',
            success: function (data) {
                // Update the energy output value in the HTML
                $('.energy-output #energy-output-value').text(data.energy_output);
            },
            error: function (xhr, status, error) {
                // Handle errors if any
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
