/**
 * creates horizontally placed cards carousel
 * @param {Array} cardsData json array
 */
function createiFrame(cardsData) {
    let cards = "";
    cardsData.map((card_item) => {
        // const item = `<div class="carousel_cards in-left">
        // <a href="${card_item.url}" target=”_blank”><img class="cardBackgroundImage" src="${card_item.image}">
        // <div class="cardFooter"> <span class="cardTitle" title="abc">${card_item.title}</span></a>
        // <div class="cardDescription"></div></div></div>`;
        const item = `<div id="container" style="position: relative;">
        <iframe src="${card_item.url}" frameborder="0" style = "width: 50vw;height: 50vh;overflow: scroll;position: absolute;left: -270px;top: -120px;"></iframe>
    </div>`
        cards += item;
    });
    const cardContents = `<div id="paginated_cards" class="cards"> <div class="cards_scroller">${cards} <span class="arrow prev fa fa-chevron-circle-left "></span> <span class="arrow next fa fa-chevron-circle-right" ></span> </div> </div>`;
    return cardContents;
}

/**
 * appends cards carousel on to the chat screen
 * @param {Array} cardsToAdd json array
 */
function showiFrame(cardsToAdd) {
    const cards = createiFrame(cardsToAdd);

    $(cards).appendTo(".chats").show();

    if (cardsToAdd.length <= 2) {
        $(`.cards_scroller>div.carousel_cards:nth-of-type(2)`).fadeIn(3000);
    } else {
        for (let i = 0; i < cardsToAdd.length; i += 1) {
            $(`.cards_scroller>div.carousel_cards:nth-of-type(${i})`).fadeIn(3000);
        }
        $(".cards .arrow.prev").fadeIn("3000");
        $(".cards .arrow.next").fadeIn("3000");
    }

    scrollToBottomOfResults();

    const card = document.querySelector("#paginated_cards");
    const card_scroller = card.querySelector(".cards_scroller");
    const card_item_size = 225;

    /**
     * For paginated scrolling, simply scroll the card one item in the given
     * direction and let css scroll snaping handle the specific alignment.
     */
    function scrollToNextPage() {
        card_scroller.scrollBy(card_item_size, 0);
    }

    function scrollToPrevPage() {
        card_scroller.scrollBy(-card_item_size, 0);
    }

    card.querySelector(".arrow.next").addEventListener("click", scrollToNextPage);
    card.querySelector(".arrow.prev").addEventListener("click", scrollToPrevPage);
    $(".usrInput").focus();
}