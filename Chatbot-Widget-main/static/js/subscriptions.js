(function() {
  fetch('/subscriptions')
    .then(response => {
      if (!response.ok) throw new Error('Failed to fetch subscription data');
      return response.json();
    })
    .then(data => populateTable(data, "subscription-table"))
    .catch(error => console.error('Error fetching subscriptions:', error));
})();
