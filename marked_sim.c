#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#define MAX_PRICE (1 << 20)             // 1_048_576
#define PAGE_SHIFT 10
#define PAGE_SIZE  (1 << PAGE_SHIFT)    // 1_024
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

typedef struct{
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

static inline uint64_t sparse_get(const SparseArray* sa, uint64_t price) {
    if (price >= MAX_PRICE) return 0;
    uint32_t page_idx = price >> PAGE_SHIFT;
    uint32_t offset   = price & (PAGE_SIZE - 1);

    if (!sa->pages[page_idx]) return 0;
    return sa->pages[page_idx][offset];
}

// Add/Set volume at price level (allocates page dynamically if missing)
static inline void sparse_add(SparseArray* sa, uint64_t price, uint64_t delta) {
    if (price >= MAX_PRICE) return;
    uint32_t page_idx = price >> PAGE_SHIFT;
    uint32_t offset   = price & (PAGE_SIZE - 1);

    if (!sa->pages[page_idx]) {
        // Allocate page using calloc so volume values start at 0
        sa->pages[page_idx] = (uint64_t*)calloc(PAGE_SIZE, sizeof(uint64_t));
    }
    sa->pages[page_idx][offset] += delta;
}

// Subtract volume at price level
static inline void sparse_sub(SparseArray* sa, uint64_t price, uint64_t delta) {
    if (price >= MAX_PRICE) return;
    uint32_t page_idx = price >> PAGE_SHIFT;
    uint32_t offset   = price & (PAGE_SIZE - 1);

    if (sa->pages[page_idx]) {
        sa->pages[page_idx][offset] -= delta;
    }
}

static inline void sync_best_prices(OrderBook* book, Side match_side) {
    if (match_side == SIDE_BID) { // We matched against asks, advance best_ask upward
        while (book->best_ask < MAX_PRICE && sparse_get(&book->asks, book->best_ask) == 0) {
            book->best_ask++;
        }
    } else { // We matched against bids, advance best_bid downward
        while (book->best_bid > 0 && sparse_get(&book->bids, book->best_bid) == 0) {
            book->best_bid--;
        }
    }
}

void process_orders(OrderBook* book) {
    for (int i = 0; i < book->num_orders; i++) {
        Order order = book->orders[i];

        if (order.price >= MAX_PRICE) continue;

        SparseArray* match_levels = (order.side == SIDE_BID) ? &book->asks : &book->bids;
        SparseArray* own_levels   = (order.side == SIDE_BID) ? &book->bids : &book->asks;

        uint64_t remaining = order.quantity;

        uint64_t price = (order.side == SIDE_BID) ? book->best_ask : book->best_bid;
        int step = (order.side == SIDE_BID) ? 1 : -1;

        // try finding offers
        while (remaining > 0) {
            // Bounds check
            if (price < 0 || price >= MAX_PRICE) break;

            if (order.type == TYPE_LIMIT) {
                if (order.side == SIDE_BID && price > order.price) break;
                if (order.side == SIDE_ASK && price < order.price) break;
            }

            uint64_t available = sparse_get(match_levels, price);

            if (available > 0) {
                if (available >= remaining) {
                    sparse_sub(match_levels, price, remaining);
                    remaining = 0;
                    // If level fully cleared, adjust top pointer
                if (available == remaining) {
                        sync_best_prices(book, order.side);
                    }
                    break;
                } else {
                    remaining -= available;
                    sparse_sub(match_levels, price, available);
                    sync_best_prices(book, order.side);
                }
            }
            price += step;
        }

        // when no offers are found add them to the book
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
    }
    book->num_orders = 0;
}

void enqueue_order(OrderBook* book, Order order) {
    if (book->num_orders >= MAX_ORDERS) return;
    memcpy(&book->orders[book->num_orders], &order, sizeof(Order));
    book->num_orders++;
}

Order create_limit_order(uint64_t price, uint64_t quantity, Side side) {
    Order order;
    order.price = price;
    order.quantity = quantity;
    order.side = side;
    order.type = TYPE_LIMIT;
    return order;
}

Order create_marked_order(uint64_t quantity, Side side) {
    Order order;
    order.price = 0;
    order.quantity = quantity;
    order.side = side;
    order.type = TYPE_MARKET;
    return order;
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
    
    // 85% Limit Orders, 15% Market Orders (Realistic liquidity provision)
    bool is_limit = (rand() % 100) < 85; 

    uint64_t quantity = (rand() % 10) + 1; // 1 to 10 units

    if (!is_limit) {
        enqueue_order(book, create_marked_order(quantity, side));
        return;
    }

    // Mid price base
    uint64_t mid_price = (book->best_ask + book->best_bid) / 2;
    if (mid_price == 0 || mid_price >= MAX_PRICE) mid_price = 1000;

    // Price offset: +/- 2% around the mid price
    int offset = (rand() % 40) - 20; // -20 to +20 ticks
    uint64_t price = mid_price + offset;

    enqueue_order(book, create_limit_order(price, quantity, side));
};

int main() {
    OrderBook book;
    memset(&book, 0, sizeof(OrderBook));
    book.best_bid = 0;
    book.best_ask = MAX_PRICE; // Essential!

    // 1. Initial Limit Orders
    enqueue_order(&book, create_limit_order(995, 40, SIDE_BID));
    enqueue_order(&book, create_limit_order(990, 20, SIDE_BID));
    enqueue_order(&book, create_limit_order(980, 4, SIDE_BID));

    enqueue_order(&book, create_limit_order(1000, 10, SIDE_ASK));
    enqueue_order(&book, create_limit_order(1010, 5, SIDE_ASK));
    enqueue_order(&book, create_limit_order(1050, 20, SIDE_ASK));

    process_orders(&book);
    print_order_book(&book, 60);

    for (int i = 0; i < 1000; i++) {
        add_random_order(&book);
    }

    process_orders(&book);
    print_order_book(&book, 60);

    return 0;
}
