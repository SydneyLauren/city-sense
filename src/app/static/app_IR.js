let get_checkboxes = function() {

    let a = $('input#chk1').prop('checked');
    let b = $('input#chk2').prop('checked');
    let c = $('input#chk3').prop('checked');
    let d = $('input#chk4').prop('checked');
    let e = $('input#chk5').prop('checked');
    let f = $('input#chk6').prop('checked');
    let g = $('input#chk7').prop('checked');
    let h = $('input#chk8').prop('checked');
    let i = $('input#chk9').prop('checked');
    let j = $('input#chk10').prop('checked');
    let k = $('input#chk11').prop('checked');
    let l = $('input#chk12').prop('checked');
    let m = $('input#chk13').prop('checked');
    let n = $('input#chk14').prop('checked');
    let o = $('input#chk15').prop('checked');
    return {'relaxing': a,
            'entertaining': b,
            'intellectual': c,
            'spiritual': d,
            'uplifting': e,
            'friendly': f,
            'profound': g,
            'creative': h,
            'surreal': i,
            'accepting': j,
            'funny': k,
            'exciting': l,
            'lively': m,
            'natural': n,
            'sophisticated': o,
            }
};

let send_checkboxes = function(cbs) {
    $.ajax({
        url: '/solve',
        contentType: "application/json; charset=utf-8",
        type: 'POST',
        success: function (data) {
            display_solutions(data);
        },
        data: JSON.stringify(cbs)
    });
};

let display_solutions = function(solutions) {
    $("span#city1").html(solutions.city_1);
    $("span#city2").html(solutions.city_2);
    $("span#city3").html(solutions.city_3);
    $("span#city4").html(solutions.city_4);
    $("span#c1text").html(solutions.c1text);
    $("span#c2text").html(solutions.c2text);
    $("span#c3text").html(solutions.c3text);
    $("span#c4text").html(solutions.c4text);
    $("img#city1pic").attr('src', solutions.city1pic);
    $("img#city2pic").attr('src', solutions.city2pic);
    $("img#city3pic").attr('src', solutions.city3pic);
    $("img#city4pic").attr('src', solutions.city4pic);
    // console.log(solutions)
    $('img#map').attr('src', solutions.map_image);
};

$(document).ready(function() {

    $("button#solve").click(function() {
        let cbs = get_checkboxes();
        send_checkboxes(cbs);
    })

})
