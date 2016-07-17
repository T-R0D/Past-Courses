/**
 *
 */

#include "marriage_game.h"

#include "stdio.h"
#include <assert.h>

// struct _mg_game_t {
//     size_t n_prop;
//     size_t n_recv;
//     uint8_t prop_prefs[MG_MAX_PLAYERS + 1];
//     uint8_t recv_prefs[MG_MAX_PLAYERS + 1];
//     uint8_t prop_matching[MG_MAX_PLAYERS];
//     uint8_t recv_matching[MG_MAX_PLAYERS];

//     int a;
//     int b;
// };


void
mg_game_test(mg_verbosity_t verbosity) {

    // const size_t n_men = 2;
    // const size_t n_women = 2;
    // uint8_t men_prefs[] = {
    //     0,1,STAY_SINGLE,
    //     0,1,STAY_SINGLE,
    // };
    // uint8_t women_prefs[] = {
    //     1,0,STAY_SINGLE,
    //     0,1,STAY_SINGLE,
    // };

    // mg_game_t* game = mg_game_new(n_men, n_women, men_prefs, women_prefs);

    // for (int i = 0; i < (n_men * (n_women + 1)); i++) {
    //     assert(game->prop_prefs[i] == men_prefs[i]);
    // }
    // for (int i = 0; i < (n_women * (n_men + 1)); i++) {
    //     assert(game->recv_prefs[i] == women_prefs[i]);
    // }

    // assert(mg_game_prop_pref(game, 0, 1) == 1);
    // assert(mg_game_prop_pref(game, 1, 0) == 0);
    // assert(mg_game_prop_pref(game, 1, 1) == 1);
    // assert(mg_game_prop_pref(game, 1, 2) == STAY_SINGLE);

    // assert(mg_game_recv_pref(game, 0, 0) == 1);
    // assert(mg_game_recv_pref(game, 0, 1) == 0);
    // assert(mg_game_recv_pref(game, 0, 2) == STAY_SINGLE);

    // mg_game_solve(game);

    // for (int i = 0; i < game->n_prop; i++) {
    //     printf("%3d", game->prop_matching[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < game->n_recv; i++) {
    //     printf("%3d", game->recv_matching[i]);
    // }
    // printf("\n");

    // mg_game_destroy(&(game));

    const size_t n_men = 5;
    const size_t n_women = 4;
    // int men_prefs[] = {
    //     1,3,0,2,4,
    //     2,3,1,0,4,
    //     2,4,1,3,0,
    //     3,2,1,4,0,
    //     1,2,3,4,0,
    // };
    // int women_prefs[] = {
    //     2,3,5,1,4,0,
    //     2,3,0,1,4,5,
    //     2,3,1,4,5,0,
    //     5,1,3,4,2,0,
    // };

    uint8_t men_prefs[] = {
        0,2,STAY_SINGLE,1,3,
        1,2,0,STAY_SINGLE,3,
        1,3,0,2,STAY_SINGLE,
        2,1,0,3,STAY_SINGLE,
        0,1,2,3,STAY_SINGLE,
    };
    uint8_t women_prefs[] = {
        1,2,4,0,3,STAY_SINGLE,
        1,2,STAY_SINGLE,0,3,4,
        1,2,0,3,4,STAY_SINGLE,
        4,0,2,3,1,STAY_SINGLE,
    };
    uint8_t man_optimal_man_matchings[] =   {2, 1, 3, STAY_SINGLE, 0};
    uint8_t man_optimal_woman_matchings[] = {4, 1, 0, 2};

    uint8_t woman_optimal_woman_matchings[] = {2, 1, 0, 4};
    uint8_t woman_optimal_man_matchings[] =   {2, 1, 0, STAY_SINGLE, 3};

    mg_game_t* game = mg_game_new(n_men, n_women, men_prefs, women_prefs);

    // for (int i = 0; i < game->n_prop; i++) {
    //     for (int j = 0; j < game->prop_pref_size; j++) {
    //         printf("%d->%d: %d\n", i, j, mg_game_prop_pref(game, i, j));
    //     }
    // }

    assert(mg_game_prop_pref(game, 0, 0) == 0);
    assert(mg_game_prop_pref(game, 0, 1) == 2);
    assert(mg_game_prop_pref(game, 0, 2) == STAY_SINGLE);
    assert(mg_game_prop_pref(game, 4, 0) == 0);

    mg_game_solve(game);
    for (int i = 0; i < game->n_prop; i++) {
        printf("%5d", game->prop_matching[i]);
        assert(game->prop_matching[i] == man_optimal_man_matchings[i]);
    }
    // printf("\n");
    for (int i = 0; i < game->n_recv; i++) {
        printf("%5d", game->recv_matching[i]);
        assert(game->recv_matching[i] == man_optimal_woman_matchings[i]);
    }
    // printf("\n");
    mg_game_destroy(&game);

    game = mg_game_new(n_women, n_men, women_prefs, men_prefs);
    mg_game_solve(game);
    for (int i = 0; i < game->n_prop; i++) {
        // printf("%5d", game->prop_matching[i]);
        assert(game->prop_matching[i] == woman_optimal_woman_matchings[i]);
    }
    // printf("\n");
    for (int i = 0; i < game->n_recv; i++) {
        // printf("%5d", game->recv_matching[i]);
        assert(game->recv_matching[i] == woman_optimal_man_matchings[i]);
    }
    // printf("\n");
}

static size_t
index(int i, int i_size, int j);


mg_game_t*
mg_game_new(size_t n_prop, size_t n_recv, uint8_t* prop_prefs, uint8_t* recv_prefs) {
    mg_game_t* self = (mg_game_t*) malloc(sizeof(mg_game_t));
    assert(self);

    assert(n_prop < MG_MAX_PLAYERS);
    assert(n_recv < MG_MAX_PLAYERS);

    self->n_prop = n_prop;
    self->n_recv = n_recv;
    self->prop_pref_size = n_recv + 1;
    self->recv_pref_size = n_prop + 1;
    for (int i = 0; i < (n_prop * (n_recv + 1)); i++) {
        self->prop_prefs[i] = prop_prefs[i];
    }
    for (int i = 0; i < (n_recv * (n_prop + 1)); i++) {
        self->recv_prefs[i] = recv_prefs[i];
    }

    return self;
}

void
mg_game_destroy(mg_game_t** self_p) {
    assert(self_p);
    if (*self_p) {
        mg_game_t* self = *self_p;
        free(self);
        *self_p = NULL;
    }
}

mg_error_t
mg_game_solve(mg_game_t* self) {
    uint8_t prop_status[MG_MAX_PLAYERS];
    uint8_t prop_next[MG_MAX_PLAYERS];
    for (int i = 0; i < self->n_prop; i++) {
        prop_status[i] = DAP_REJECTED;
        prop_next[i] = 0;
    }
    uint8_t tentative_acceptances[MG_MAX_PLAYERS];
    for (int i = 0; i < self->n_recv; i++) {
        tentative_acceptances[i] = STAY_SINGLE;
    }

    uint8_t n_rejections = self->n_prop;
    while (n_rejections > 0) {
        n_rejections = 0;

        for (int i = 0; i < self->n_prop; i++) {
            
            printf("\nProposer %d\n----------------\n", i);

            if (prop_status[i] == DAP_REJECTED) {
                uint8_t j = mg_game_prop_pref(self, i, prop_next[i]);
                printf("\tpreference = %d\n", j);

                if (j == STAY_SINGLE) {
                    printf("\topting to SS\n");

                    prop_status[i] = DAP_STAYING_SINGLE;
                } else {
                    printf("\tpropose to %d\n", j);

                    for (int k = 0; k < self->recv_pref_size; k++) {
                        uint8_t receiver_pref = mg_game_recv_pref(self, j, k);
                        uint8_t tentative_pref = tentative_acceptances[j];


                        printf("\t\trecv_pref: %d\ttentative_pref: %d\n", receiver_pref, tentative_pref);


                        if (receiver_pref == i) {
                            printf("\t%d accepts %d's proposal, rejects %d\n", j, i, tentative_pref);

                            tentative_acceptances[j] = i;
                            prop_status[i] = DAP_TENTATIVELY_ACCEPTED;
                            if (tentative_pref != STAY_SINGLE) {
                                prop_status[tentative_pref] = DAP_REJECTED;
                                prop_next[tentative_pref]++;
                            }

                            break;
                        } else if (receiver_pref == tentative_pref) {
                            printf("\t%d rejects %d's proposal (stays with %d)\n", j, i, tentative_pref);

                            prop_status[i] = DAP_REJECTED;
                            prop_next[i]++;

                            break;
                        }
                    }
                }

            } else {
                printf("\ttentatively accepted - do nothing\n");
            }
        }

        for (int i = 0; i < self->n_prop; i++) {
            if (prop_status[i] == DAP_REJECTED) {
                n_rejections++;
            }
        }

        printf("N-rejections: %d\n", n_rejections);
    }

    for (int i = 0; i < self->n_prop; i++) {
        if (prop_status[i] == DAP_STAYING_SINGLE) {
            self->prop_matching[i] = STAY_SINGLE;
        }
    }
    for (int j = 0; j < self->n_recv; j++) {
        uint8_t i = tentative_acceptances[j];
        self->recv_matching[j] = i;
        self->prop_matching[i] = j;
    }


    return MG_SUCCESS;
}

uint8_t
mg_game_prop_pref(mg_game_t* self, uint8_t proposer, uint8_t pref_rank) {
    assert(self);
    int index = (proposer * self->prop_pref_size) + pref_rank;
    return self->prop_prefs[index];
}

uint8_t
mg_game_recv_pref(mg_game_t* self, uint8_t receiver, uint8_t pref_rank) {
    assert(self);
    int index = (receiver * self->recv_pref_size) + pref_rank;
    return self->recv_prefs[index];
}

uint8_t
mg_game_prop_matchings() {

}

uint8_t
mg_game_recv_matchings() {

}

static size_t
index(int i, int i_size, int j) {
    return (i * i_size) + j;
}
