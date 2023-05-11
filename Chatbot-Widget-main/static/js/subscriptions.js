
document.addEventListener('DOMContentLoaded', () => {
    fetch('http://localhost:3000/subscriptions')
      .then(response => {
        if (response.status !== 200) {
          throw new Error('Failed to fetch subscription data');
        }
        return response.json();
      })
      .then(subscriptionData => {
        console.log(subscriptionData);
        populateTable(subscriptionData, "subscription-table");
      })
      .catch(error => {
        console.error('Error fetching subscription data:', error);
      });
  });