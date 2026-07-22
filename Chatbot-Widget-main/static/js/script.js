/* Module for importing other js files */
function include(file) {
  const script = document.createElement('script');
  script.src = file;
  script.type = 'text/javascript';
  script.defer = true;
  document.getElementsByTagName('head').item(0).appendChild(script);
}

/* Populate an HTML table from JSON data */
function populateTable(data, tableId) {
  const table = document.getElementById(tableId);
  if (!data || data.length === 0) return;

  const keys = Object.keys(data[0]).filter(k => k !== '_id' && k !== '__v');

  // Header
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  keys.forEach(key => {
    const th = document.createElement('th');
    th.textContent = key.replace(/_/g, ' ').toUpperCase();
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  // Body
  const tbody = document.createElement('tbody');
  data.forEach(obj => {
    const row = document.createElement('tr');
    keys.forEach(key => {
      const cell = document.createElement('td');
      cell.textContent = obj[key] ?? '';
      row.appendChild(cell);
    });
    tbody.appendChild(row);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
}

// Bot pop-up intro
document.addEventListener("DOMContentLoaded", () => {
  const elemsTap = document.querySelector(".tap-target");
  if (elemsTap) {
    const instancesTap = M.TapTarget.init(elemsTap, {});
    instancesTap.open();
    setTimeout(() => instancesTap.close(), 4000);
  }
});

/* Import components */
include('./static/js/components/index.js');

window.addEventListener('load', () => {
  $(document).ready(() => {
    $("div").removeClass("tap-target-origin");
    $(".dropdown-trigger").dropdown();
    $(".modal").modal();
  });

  // Toggle chatbot widget
  $("#profile_div").click(() => {
    $(".profile_div").toggle();
    $(".widget").toggle();
  });

  // Minimize widget by clicking header
  $(".chat_header_title").click(() => {
    $(".profile_div").toggle();
    $(".widget").toggle();
  });

  // Clear chat
  $("#clear").click(() => {
    $(".chats").fadeOut("normal", () => {
      $(".chats").html("");
      $(".chats").fadeIn();
    });
  });

  // Close widget
  $("#close").click(() => {
    $(".profile_div").toggle();
    $(".widget").toggle();
    scrollToBottomOfResults();
  });
});
