/* Quiz component — immediate-feedback multiple choice for retrieval practice.
 *
 * Markup contract:
 *   <div class="quiz" data-answer="1">
 *     <p class="quiz-q">Question text</p>
 *     <pre class="quiz-stem">optional monospace stem</pre>
 *     <div class="quiz-opts">
 *       <button class="quiz-opt" data-fb="Shown when this option is picked.">Option one</button>
 *       <button class="quiz-opt" data-fb="...">Option two</button>
 *     </div>
 *     <div class="quiz-fb"><p class="quiz-why">Always-shown explanation after answering.</p></div>
 *   </div>
 *
 * data-answer is the 0-indexed correct option.
 * Per-option data-fb is optional; the .quiz-why paragraph always shows.
 */
(function () {
  function init(quiz) {
    var answer = parseInt(quiz.dataset.answer, 10);
    var opts = Array.prototype.slice.call(quiz.querySelectorAll('.quiz-opt'));
    var fb = quiz.querySelector('.quiz-fb');
    if (!fb) return;

    var why = fb.querySelector('.quiz-why');
    var verdict = document.createElement('p');
    verdict.className = 'verdict-line';
    fb.insertBefore(verdict, fb.firstChild);

    opts.forEach(function (opt, i) {
      opt.addEventListener('click', function () {
        if (quiz.dataset.done) return;
        quiz.dataset.done = '1';

        var right = i === answer;
        opts.forEach(function (o, j) {
          o.disabled = true;
          if (j === answer) o.classList.add('is-correct');
          else if (j === i) o.classList.add('is-wrong');
          else o.classList.add('is-muted');
        });

        var picked = opt.dataset.fb ? ' ' + opt.dataset.fb : '';
        verdict.innerHTML =
          '<span class="verdict ' + (right ? 'ok' : 'no') + '">' +
          (right ? 'Correct.' : 'Not quite.') + '</span>' + picked;

        fb.classList.add('show');
        if (why) why.style.display = 'block';
      });
    });
  }

  function boot() {
    document.querySelectorAll('.quiz').forEach(init);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
