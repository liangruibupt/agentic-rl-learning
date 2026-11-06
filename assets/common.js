// Shared navigation, pager, segmented buttons and quiz helper.
const PAGES = [
  { f: 'index.html', t: '学习路线图', lv: '' },
  { f: '01-basics-what-is-agentic-rl.html', t: '01 什么是 Agentic RL', lv: '基础班' },
  { f: '02-basics-training-loop.html', t: '02 一次训练迭代', lv: '基础班' },
  { f: '03-basics-grpo.html', t: '03 GRPO 入门', lv: '基础班' },
  { f: '04-intermediate-reward-design.html', t: '04 奖励设计', lv: '中级班' },
  { f: '05-intermediate-agentic-techniques.html', t: '05 多轮训练关键技术', lv: '中级班' },
  { f: '06-advanced-algorithms.html', t: '06 算法家族', lv: '进阶班' },
  { f: '07-advanced-systems.html', t: '07 系统架构与异步', lv: '进阶班' },
  { f: '08-landscape.html', t: '08 前沿玩家与方案', lv: '进阶班' },
  { f: '09-build-your-own.html', t: '09 客户自建', lv: '落地班' },
  { f: '10-aws-experiment.html', t: '10 AWS 实验设计', lv: '落地班' },
];
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
function seg(btns, onSel) {
  btns.forEach(b => b.addEventListener('click', () => {
    btns.forEach(x => x.classList.remove('on')); b.classList.add('on'); onSel(b);
  }));
}
(function buildNav() {
  const here = location.pathname.split('/').pop() || 'index.html';
  const nav = document.createElement('nav');
  nav.className = 'side';
  let h = '<a class="brand" href="index.html">Agentic RL 学习课程</a>';
  let lastLv = null;
  PAGES.forEach(p => {
    if (p.lv && p.lv !== lastLv) { h += `<div class="lv">${p.lv}</div>`; lastLv = p.lv; }
    h += `<a class="pg${p.f === here ? ' on' : ''}" href="${p.f}">${p.t}</a>`;
  });
  nav.innerHTML = h;
  document.body.prepend(nav);
  const main = document.querySelector('main');
  const i = PAGES.findIndex(p => p.f === here);
  if (main && i >= 0) {
    const prev = PAGES[i - 1], next = PAGES[i + 1];
    const pg = document.createElement('div');
    pg.className = 'pager';
    pg.innerHTML = (prev ? `<a href="${prev.f}">← ${prev.t}</a>` : '<span></span>') +
                   (next ? `<a href="${next.f}">${next.t} →</a>` : '<span></span>');
    main.appendChild(pg);
  }
})();
// Quiz: <div class="quiz" data-answer="1"><b>Q</b><button class="btn opt">..</button>...<div class="qe">explain</div></div>
document.addEventListener('DOMContentLoaded', () => {
  $$('.quiz').forEach(q => {
    const ans = +q.dataset.answer;
    const ex = q.querySelector('.qe');
    if (ex) ex.style.display = 'none';
    q.querySelectorAll('.opt').forEach((o, i) => o.addEventListener('click', () => {
      q.querySelectorAll('.opt').forEach(x => x.classList.remove('right', 'wrong'));
      o.classList.add(i === ans ? 'right' : 'wrong');
      if (ex) ex.style.display = 'block';
    }));
  });
});
