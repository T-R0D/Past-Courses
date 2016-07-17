/**
 *
 */

#ifndef _MARRIAGE_GAME
#define _MARRIAGE_GAME 1

#include <stdlib.h>
#include <stdint.h>
#include <assert.h>


#define MG_MAX_PLAYERS 300

#define STAY_SINGLE 255

typedef enum mg_verbosity_t {
    MG_QUIET = 0,
    MG_VERBOSE,
    MG_VERBOSE_MAX,
} mg_verbosity_t;

typedef enum mg_error_t {
    MG_SUCCESS = 0,
    MG_OVERFLOW_ERROR,
} mg_error_t;

typedef enum mg_dapstatus_t {
    DAP_REJECTED,
    DAP_TENTATIVELY_ACCEPTED,
    DAP_STAYING_SINGLE,
} DapStatus;

typedef struct _mg_game_t {
    size_t n_prop;
    size_t n_recv;
    uint8_t prop_prefs[MG_MAX_PLAYERS + 1];
    uint8_t recv_prefs[MG_MAX_PLAYERS + 1];
    uint8_t prop_matching[MG_MAX_PLAYERS];
    uint8_t recv_matching[MG_MAX_PLAYERS];

    size_t prop_pref_size;
    size_t recv_pref_size;
} mg_game_t;


mg_game_t*
    mg_game_new(size_t n_prop, size_t n_recv, uint8_t* prop_prefs, uint8_t* recv_prefs);

void
    mg_game_destroy(mg_game_t** self);

mg_error_t
    mg_game_solve(mg_game_t* self);

uint8_t
    mg_game_prop_pref(mg_game_t*self, uint8_t proposer, uint8_t pref_rank);

uint8_t
    mg_game_recv_pref(mg_game_t* self, uint8_t receiver, uint8_t pref_rank);

uint8_t
    mg_game_prop_matchings();

uint8_t
    mg_game_recv_matchings();

void
    mg_game_test(mg_verbosity_t verbosity);


#endif
