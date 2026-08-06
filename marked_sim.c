#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#define MAX_PRICE (1 << 20)           // 1_048_576
#define PAGE_SHIFT 10
#define PAGE_SIZE  (1 << PAGE_SHIFT)  // 1_024
#define NUM_PAGES  ((MAX_PRICE + PAGE_SIZE - 1) / PAGE_SIZE)

#define MAX_ORDERS 1000

typedef enum {
    SIDE_BID = 0,
    SIDE_ASK = 1,
} Side;

typedef enum {
    TYPE_LIMIT = 0,
    TYPE_MARKET = 1,
} OrderType;

typedef struct {
    uint64_t price;
    uint64_t quantity;
    Side side;
    OrderType type;
} Order;

typedef struct {
    uint64_t* pages[NUM_PAGES];
} SparseArray;

typedef struct {
    SparseArray asks;
    SparseArray bids;

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

static inline uint64_t sparse_get(const SparseArray* sa, uint64_t price) {
    if (price >= MAX_PRICE) return 0;
    uint32_t page_idx = price >> PAGE_SHIFT;
    uint32_t offset   = price & (PAGE_SIZE - 1);

    if (!sa->pages[page_idx]) return 0;
    return sa->pages[page_idx][offset];
}

static inline void sparse_add(SparseArray* sa, uint64_t price, uint64_t delta) {
    if (price >= MAX_PRICE) return;
    uint32_t page_idx = price >> PAGE_SHIFT;
    uint32_t offset   = price & (PAGE_SIZE - 1);

    if (!sa->pages[page_idx]) {
        sa->pages[page_idx] = (uint64_t*)calloc(PAGE_SIZE, sizeof(uint64_t));
    }
    sa->pages[page_idx][offset] += delta;
}

static inline void sparse_sub(SparseArray* sa, uint64_t price, uint64_t delta) {
    if (price >= MAX_PRICE) return;
    uint32_t page_idx = price >> PAGE_SHIFT;
    uint32_t offset   = price & (PAGE_SIZE - 1);

    if (sa->pages[page_idx]) {
        sa->pages[page_idx][offset] -= delta;
    }
}

static inline void sync_best_prices(OrderBook* book) {
    while (book->best_ask < MAX_PRICE && sparse_get(&book->asks, book->best_ask) == 0) {
        book->best_ask++;
    }
    while (book->best_bid > 0 && sparse_get(&book->bids, book->best_bid) == 0) {
        book->best_bid--;
    }
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

        SparseArray* match_levels = (order.side == SIDE_BID) ? &book->asks : &book->bids;
        SparseArray* own_levels   = (order.side == SIDE_BID) ? &book->bids : &book->asks;

        uint64_t remaining = order.quantity;
        uint64_t price = (order.side == SIDE_BID) ? book->best_ask : book->best_bid;
        int step = (order.side == SIDE_BID) ? 1 : -1;

        while (remaining > 0) {
            if (price == 0 || price >= MAX_PRICE) break;

            if (order.type == TYPE_LIMIT) {
                if (order.side == SIDE_BID && price > order.price) break;
                if (order.side == SIDE_ASK && price < order.price) break;
            }

            uint64_t available = sparse_get(match_levels, price);

            if (available > 0) {
                uint64_t fill = (available >= remaining) ? remaining : available;
                sparse_sub(match_levels, price, fill);
                remaining -= fill;

                // Record execution price and matched quantity into the active candle
                record_trade(current_bar, price, fill);
            }
            price += step;
        }

        if (order.type == TYPE_LIMIT && remaining > 0) {
            sparse_add(own_levels, order.price, remaining);

            if (order.side == SIDE_BID) {
                if (book->best_bid == 0 || order.price > book->best_bid) {
                    book->best_bid = order.price;
                }
            } else {
                if (book->best_ask == MAX_PRICE || order.price < book->best_ask) {
                    book->best_ask = order.price;
                }
            }
        }
        sync_best_prices(book);
    }
    book->num_orders = 0;
}

void enqueue_order(OrderBook* book, Order order) {
    if (book->num_orders >= MAX_ORDERS) return;
    memcpy(&book->orders[book->num_orders], &order, sizeof(Order));
    book->num_orders++;
}

Order create_limit_order(uint64_t price, uint64_t quantity, Side side) {
    return (Order){ .price = price, .quantity = quantity, .side = side, .type = TYPE_LIMIT };
}

Order create_market_order(uint64_t quantity, Side side) {
    return (Order){ .price = 0, .quantity = quantity, .side = side, .type = TYPE_MARKET };
}


void print_order_book(const OrderBook* book, uint32_t range) {
    uint64_t center = (book->best_ask + book->best_bid ) / 2;
    uint64_t start_price = (center > range) ? (center - range) : 0;
    uint64_t end_price   = (center + range < MAX_PRICE) ? (center + range) : (MAX_PRICE - 1);

    printf("\n=========================================\n");
    printf("         ORDER BOOK (b_aks: %i b_bid: %i)        \n", book->best_ask, book->best_bid);
    printf("=========================================\n");
    printf("   Price   |   Ask Vol   |   Bid Vol    \n");
    printf("-----------------------------------------\n");

    bool found_asks = false;
    for (int64_t p = (int64_t)end_price; p >= (int64_t)center; p--) {
        uint64_t vol = sparse_get(&book->asks, p);
        if (vol > 0) {
            printf("  $%6lld | %11llu |              \n", (long long)p, (unsigned long long)vol);
            found_asks = true;
        }
    }
    if (!found_asks) printf("  [ No Asks in range ]                  \n");

    printf("---------- CURRENT PRICE LEVEL %4llu ----------\n", (unsigned long long)center);

    bool found_bids = false;
    for (int64_t p = (int64_t)center; p >= (int64_t)start_price; p--) {
        uint64_t vol = sparse_get(&book->bids, p);
        if (vol > 0) {
            printf("  $%6lld |              | %11llu  \n", (long long)p, (unsigned long long)vol);
            found_bids = true;
        }
    }
    if (!found_bids) printf("  [ No Bids in range ]                  \n");

    printf("=========================================\n\n");
}
void add_random_order(OrderBook* book) {
    Side side = (rand() % 2 == 0) ? SIDE_BID : SIDE_ASK;
    
    // 70% Limit, 30% Market Orders to promote trade matches for volume
    bool is_limit = (rand() % 100) < 70; 
    uint64_t quantity = (rand() % 15) + 1;

    if (!is_limit) {
        enqueue_order(book, create_market_order(quantity, side));
        return;
    }

    uint64_t mid_price = (book->best_ask + book->best_bid) / 2;
    if (mid_price == 0 || mid_price >= MAX_PRICE) mid_price = 1000;

    int offset = (rand() % 20) - 10; // Tight spread (-10 to +10)
    uint64_t price = mid_price + offset;

    enqueue_order(book, create_limit_order(price, quantity, side));
}

int main() {
    OrderBook book;
    memset(&book, 0, sizeof(OrderBook));
    book.best_bid = 0;
    book.best_ask = MAX_PRICE;

    // Initial Depth Setup
    enqueue_order(&book, create_limit_order(995, 100, SIDE_BID));
    enqueue_order(&book, create_limit_order(990, 100, SIDE_BID));
    enqueue_order(&book, create_limit_order(1000, 100, SIDE_ASK));
    enqueue_order(&book, create_limit_order(1005, 100, SIDE_ASK));

    OHLCVBar init_dummy = {0};
    process_orders(&book, &init_dummy);

    int total_bars = 5;
    int orders_per_bar = 50;

    printf("=========================================================\n");
    printf(" Bar |    Open |    High |     Low |   Close |   Volume  \n");
    printf("=========================================================\n");

    for (int bar_idx = 1; bar_idx <= total_bars; bar_idx++) {
        OHLCVBar bar = {0};

        for (int i = 0; i < orders_per_bar; i++) {
            add_random_order(&book);
            process_orders(&book, &bar);
        }

        if (bar.has_trades) {
            printf(" %3d | %7llu | %7llu | %7llu | %7llu | %8llu\n",
                bar_idx,
                (unsigned long long)bar.open,
                (unsigned long long)bar.high,
                (unsigned long long)bar.low,
                (unsigned long long)bar.close,
                (unsigned long long)bar.volume);
        } else {
            printf(" %3d |   NO TRADES EXECUTED THIS PERIOD\n", bar_idx);
        }
    }
    printf("=========================================================\n");

    return 0;
}
