document.addEventListener('DOMContentLoaded', () => {
  fetch(`${backend_url}/customers`)
    .then(response => {
      if (!response.ok) throw new Error('Failed to fetch customer data');
      return response.json();
    })
    .then(data => populateTable(data, "customers-table"))
    .catch(error => console.error('Error fetching customers:', error));
});
