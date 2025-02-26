/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

/*
 *   Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License").
 *   You may not use this file except in compliance with the License.
 *   A copy of the License is located at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   or in the "license" file accompanying this file. This file is distributed
 *   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *   express or implied. See the License for the specific language governing
 *   permissions and limitations under the License.
 */

package com.amazon.opendistroforelasticsearch.sql.legacy.unittest.metrics;

import com.amazon.opendistroforelasticsearch.sql.legacy.metrics.BasicCounter;
import org.junit.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;

public class BasicCounterTest {

    @Test
    public void increment() {
        BasicCounter counter = new BasicCounter();
        for (int i=0; i<5; ++i) {
            counter.increment();
        }

        assertThat(counter.getValue(), equalTo(5L));
    }

    @Test
    public void incrementN() {
        BasicCounter counter = new BasicCounter();
        counter.add(5);

        assertThat(counter.getValue(), equalTo(5L));
    }

}
