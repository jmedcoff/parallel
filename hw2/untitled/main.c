#include <stdio.h>
#include <stdlib.h>


/* -------------------------------- */
/* ----------- List ops ----------- */
/* -------------------------------- */

typedef struct node {
    void *data; //generic data
    struct node *next; //successor node
} node;

node* node_create(void *data, node *next) {
    node* newnode = (node*)malloc(sizeof(node));
    newnode->data = data;
    newnode->next = next;
    return newnode;
}

node* node_push(node* head, void *data) {
    node* newnode = node_create(data, head);
    head = newnode;
    return head;
}

node* node_append(node* head, void *data) {
    node* current = head;
    while (current->next != NULL)
        current = current->next;
    node* newnode = node_create(data, NULL);
    current->next = newnode;
    return head;
}

node* node_search(node* head, void *data) {
    node* current = head;
    while (current != NULL) {
        if (current->data == data)
            return current;
        current = current->next;
    }
    return NULL;
}

void node_dispose(node* head) {
    node *current, *temp;
    if (head != NULL) {
        current = head->next;
        head->next = NULL;
        while (current != NULL) {
            temp = current->next;
            free(current);
            current = temp;
        }
    }
}

/* -------------------------------- */
/* ---------- Hash table ---------- */
/* -------------------------------- */

typedef struct entry {
    unsigned int key;
    node* values;
} entry;

typedef struct hashtable {
    int size;
    int num_entries;
    entry* table[];
} hashtable;

// init table with keys=0 and no values
hashtable* create_table(int n) {
    entry* tab = malloc(n*sizeof(entry));
    hashtable* newtable = malloc(sizeof(hashtable));
    newtable->size = n;
    for (int i=0; i<n; i++) {
        newtable->table[i]->key = 0;
        newtable->table[i]->values = NULL;
    }
    return newtable;
}

void free_table(hashtable* h) {
    for (int i=0; i<h->size; i++) {
        h->table[i]->values = NULL;
    }
    free(h->table);
    free(h);
}

// TODO: Come up with something intelligent
unsigned int hash_state(int a[4][4]) {
    return 0;
}

void add_to_table(hashtable* h, int a[4][4]) {
    unsigned int key = hash_state(a);
    h->table[key]->key = key;
    node_append(h->table[key]->values, a);
}

int check_table(hashtable* h, int a[4][4]) {
    unsigned int key = hash_state(a);
    node* res = node_search(h->table[key]->values, a);
    if (res)
        return 1;
    return 0;
}





/* -------------------------------- */
/* ---------- Puzzle game --------- */
/* -------------------------------- */

//the finished game. 0 represents the empty tile
const int done[4][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 0}};

int is_done(int s[4][4]) {
    int i, j;
    for (i=0; i<4; i++) {
        for (j=0; j<4; j++) {
            if (s[i][j] != done[i][j])
                return 0; // not done
        }
    }
    return 1; // done
}

//movement controls: designate motion of the empty tile

void left(int s[4][4]) {
    int i, j, temp;
    for (i=0; i<4; i++) {
        for (j=0; j<4; j++) {
            if (s[i][j] == 0 && j != 0) {
                temp = s[i][j-1];
                s[i][j-1] = 0;
                s[i][j] = temp;
                return;
            }
        }
    }
}

void right(int s[4][4]) {
    int i, j, temp;
    for (i=0; i<4; i++) {
        for (j=0; j<4; j++) {
            if (s[i][j] == 0 && j != 3) {
                temp = s[i][j+1];
                s[i][j+1] = 0;
                s[i][j] = temp;
                return;
            }
        }
    }
}

void up(int s[4][4]) {
    int i, j, temp;
    for (i=0; i<4; i++) {
        for (j=0; j<4; j++) {
            if (s[i][j] == 0 && i != 0) {
                temp = s[i-1][j];
                s[i-1][j] = 0;
                s[i][j] = temp;
                return;
            }
        }
    }
}

void down(int s[4][4]) {
    int i, j, temp;
    for (i=0; i<4; i++) {
        for (j=0; j<4; j++) {
            if (s[i][j] == 0 && i != 3) {
                temp = s[i+1][j];
                s[i+1][j] = 0;
                s[i][j] = temp;
                return;
            }
        }
    }
}



int main() {
    int new[4][4] = {
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 0, 12},
            {13, 14, 11, 15}};
    down(new);
    int i, j;
    for (i=0; i<4; i++) {
        for (j = 0; j<4; j++) {
            printf("%d ", new[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    return 0;
}