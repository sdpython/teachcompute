#include "teachcompute_test_common.h"
#include "teachcompute_helpers.h"
#include "teachcompute/validation/cpu/vector_function.h"

using namespace validation;

void testAssertTrue() {
  ASSERT_THROW( true );
}

void test_vector_sum() {
    std::vector<float> m(10);
    for(std::size_t i = 0; i < m.size(); ++i) {
        m[i] = static_cast<float>(i);
    }
    float s1 = vector_sum(2, 5, &m[0], true);
    float s2 = vector_sum(2, 5, &m[0], false);
    ASSERT_THROW(s1 == s2);
}

// from test_cpu_fpemu.cpp
void test_cast();

int main(int, char**) {
  testAssertTrue();
  test_vector_sum();
  // from test_cpu_fpemu.cpp
  test_cast();
}
