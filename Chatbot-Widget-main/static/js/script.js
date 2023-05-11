
/* module for importing other js files */
function include(file) {
  const script = document.createElement('script');
  script.src = file;
  script.type = 'text/javascript';
  script.defer = true;

  document.getElementsByTagName('head').item(0).appendChild(script);
}


function populateTable(data, tableId) {
  const table = document.getElementById(tableId);
  
  // Create the table header
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  
  // Extract the keys from the first object in the data
  const keys = Object.keys(data[0]);
  
  // Create table headings based on the keys
  keys.forEach((key) => {
    const formattedKey = key.replace(/_/g, ' '); // Remove underscores
    const capitalizedHeading = formattedKey.toUpperCase();
    const th = document.createElement('th');
    th.textContent = capitalizedHeading;
    headerRow.appendChild(th);
  });
  console.log(table);
  
  thead.appendChild(headerRow);

  
  // Create the table body
  const tbody = document.createElement('tbody');
  
  // Populate the table with data
  data.forEach((obj) => {
    const row = document.createElement('tr');
    
    keys.forEach((key) => {
      const cell = document.createElement('td');
      cell.textContent = obj[key];
      row.appendChild(cell);
    });
    
    tbody.appendChild(row);
  });
  console.log(tbody);
  table.appendChild(thead);
  table.appendChild(tbody);
}

// 
// Bot pop-up intro
document.addEventListener("DOMContentLoaded", () => {
  const elemsTap = document.querySelector(".tap-target");
  // eslint-disable-next-line no-undef
  const instancesTap = M.TapTarget.init(elemsTap, {});
  instancesTap.open();
  setTimeout(() => {
    instancesTap.close();
  }, 4000);
});

/* import components */
include('./static/js/components/index.js');

window.addEventListener('load', () => {
  // initialization
  $(document).ready(() => {
    // Bot pop-up intro
    $("div").removeClass("tap-target-origin");

    // drop down menu for close, restart conversation & clear the chats.
    $(".dropdown-trigger").dropdown();

    // initiate the modal for displaying the charts,
    // if you dont have charts, then you comment the below line
    $(".modal").modal();

    // enable this if u have configured the bot to start the conversation.
    // showBotTyping();
    // $("#userInput").prop('disabled', true);

    // if you want the bot to start the conversation
    // customActionTrigger();
  });
  // Toggle the chatbot screen
  $("#profile_div").click(() => {
    $(".profile_div").toggle();
    $(".widget").toggle();
  });

  // clear function to clear the chat contents of the widget.
  $("#clear").click(() => {
    $(".chats").fadeOut("normal", () => {
      $(".chats").html("");
      $(".chats").fadeIn();
    });
  });

  // close function to close the widget.
  $("#close").click(() => {
    $(".profile_div").toggle();
    $(".widget").toggle();
    scrollToBottomOfResults();
  });
});
