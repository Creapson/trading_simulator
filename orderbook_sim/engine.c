#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define MAX_PRICE (1 << 20)
#define MAX_ORDERS 1000

typedef enum { SIDE_BID = 0, SIDE_ASK = 1 } Side;
typedef enum { TYPE_LIMIT = 0, TYPE_MARKET = 1 } OrderType;

typedef struct {
    uint64_t price;
    uint64_t quantity;
    Side side;
    OrderType type;
} Order;

typedef struct {
    uint64_t* asks;
    uint64_t* bids;
    Order orders[MAX_ORDERS];
    uint32_t num_orders;
    uint32_t best_ask;
    uint32_t best_bid;
} OrderBook;

typedef struct {
    uint64_t open;
    uint64_t high;
    uint64_t low;
    uint64_t close;
    uint64_t volume;
    bool has_trades;
} OHLCVBar;

static inline void sync_best_prices(OrderBook* book) {
    while (book->best_ask < MAX_PRICE && book->asks[book->best_ask] == 0) book->best_ask++;
    while (book->best_bid > 0 && book->bids[book->best_bid] == 0) book->best_bid--;
}

void record_trade(OHLCVBar* bar, uint64_t price, uint64_t qty) {
    if (!bar->has_trades) {
        bar->open = price;
        bar->high = price;
        bar->low = price;
        bar->close = price;
        bar->volume = qty;
        bar->has_trades = true;
    } else {
        if (price > bar->high) bar->high = price;
        if (price < bar->low)  bar->low = price;
        bar->close = price;
        bar->volume += qty;
    }
}

void process_orders(OrderBook* book, OHLCVBar* current_bar) {
    for (int i = 0; i < book->num_orders; i++) {
        Order order = book->orders[i];
        if (order.price >= MAX_PRICE && order.type == TYPE_LIMIT) continue;

        uint64_t* match_levels = (order.side == SIDE_BID) ? book->asks : book->bids;
        uint64_t* own_levels   = (order.side == SIDE_BID) ? book->bids : book->asks;

        uint64_t remaining = order.quantity;
        uint64_t price = (order.side == SIDE_BID) ? book->best_ask : book->best_bid;
        int step = (order.side == SIDE_BID) ? 1 : -1;

        while (remaining > 0) {
            if (price == 0 || price >= MAX_PRICE) break;

            if (order.type == TYPE_LIMIT) {
                if (order.side == SIDE_BID && price > order.price) break;
                if (order.side == SIDE_ASK && price < order.price) break;
            }

            uint64_t available = match_levels[price];

            if (available > 0) {
                uint64_t fill = (available >= remaining) ? remaining : available;
                match_levels[price] -= fill;
                remaining -= fill;
                record_trade(current_bar, price, fill);
            }
            price += step;
        }

        if (order.type == TYPE_LIMIT && remaining > 0) {
            own_levels[order.price] += remaining;

            if (order.side == SIDE_BID) {
                if (book->best_bid == 0 || order.price > book->best_bid) book->best_bid = order.price;
            } else {
                if (book->best_ask == MAX_PRICE || order.price < book->best_ask) book->best_ask = order.price;
            }
        }
        sync_best_prices(book);
    }
    book->num_orders = 0;
}

void enqueue_order(OrderBook* book, uint64_t price, uint64_t quantity, Side side, OrderType type) {
    if (book->num_orders >= MAX_ORDERS) return;
    book->orders[book->num_orders] = (Order){ .price = price, .quantity = quantity, .side = side, .type = type };
    book->num_orders++;
}

void add_random_order(OrderBook* book, uint64_t last_traded_price) {
    Side side = (rand() % 2 == 0) ? SIDE_BID : SIDE_ASK;
    double yoy_growth = 0.07;
    int trading_days_per_year = 252;

    bool is_limit = (rand() % 100) < 20; 
    uint64_t quantity = (rand() % 50) + 1;

    if (!is_limit) {
        enqueue_order(book, 0, quantity, side, TYPE_MARKET);
        return;
    }

    double price = last_traded_price;
    if (book->best_bid > 0 && book->best_ask < MAX_PRICE && book->best_ask > book->best_bid) {
        price = (book->best_ask + book->best_bid) / 2.0;
    }

    double y = (((double)rand() / RAND_MAX) * 0.02) + 1;
    bool is_pos = (rand() % 2 == 1);

    if (is_pos) { price *= (y + yoy_growth/trading_days_per_year); }
    else { price *= (1/y); }
    price += 0.5;

    enqueue_order(book, (uint64_t)price, quantity, side, TYPE_LIMIT);
}

OrderBook* init_orderbook() {
    OrderBook* book = (OrderBook*)calloc(1, sizeof(OrderBook));
    book->asks = (uint64_t*)calloc(MAX_PRICE, sizeof(uint64_t));
    book->bids = (uint64_t*)calloc(MAX_PRICE, sizeof(uint64_t));
    book->best_bid = 0;
    book->best_ask = MAX_PRICE;
    return book;
}

void free_orderbook(OrderBook* book) {
    if (!book) return;
    if (book->asks) free(book->asks);
    if (book->bids) free(book->bids);
    free(book);
}

// Full simulation routine called directly from Python
int run_simulation_c(int total_bars, const char* output_filename) {
    srand(time(NULL));

    OrderBook* book = init_orderbook();

    enqueue_order(book, 9950, 100, SIDE_BID, TYPE_LIMIT);
    enqueue_order(book, 9900, 100, SIDE_BID, TYPE_LIMIT);
    enqueue_order(book, 10000, 100, SIDE_ASK, TYPE_LIMIT);
    enqueue_order(book, 10050, 100, SIDE_ASK, TYPE_LIMIT);

    OHLCVBar init_dummy = {0};
    process_orders(book, &init_dummy);

    FILE* csv_file = fopen(output_filename, "w");
    if (!csv_file) {
        free_orderbook(book);
        return -1;
    }

    fprintf(csv_file, "Date,OPEN,HIGH,LOW,CLOSE,VOLUME\n");

    uint64_t last_price = 10000;

    for (int bar_idx = 1; bar_idx <= total_bars; bar_idx++) {
        OHLCVBar bar = {0};
        int orders_per_bar = rand() % 100 + 10;
        for (int i = 0; i < orders_per_bar; i++) {
            add_random_order(book, last_price);
            process_orders(book, &bar);
        }

        if (bar.has_trades) {
            last_price = bar.close;
            fprintf(csv_file, "%d,%llu,%llu,%llu,%llu,%llu\n",
                bar_idx,
                (unsigned long long)bar.open,
                (unsigned long long)bar.high,
                (unsigned long long)bar.low,
                (unsigned long long)bar.close,
                (unsigned long long)bar.volume);
        }
    }

    fclose(csv_file);
    free_orderbook(book);
    return 0;
}
