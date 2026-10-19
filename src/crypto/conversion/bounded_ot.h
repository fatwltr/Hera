#pragma once

// SCI's NetIO header requires its namespace constants to be defined before
// EMP/SilentOT includes it indirectly. Keep that dependency order in one place.
#include "utils/constants.h"
#include "utils/io_pack.h"
#include "OT/ferret/silent_ot.h"
