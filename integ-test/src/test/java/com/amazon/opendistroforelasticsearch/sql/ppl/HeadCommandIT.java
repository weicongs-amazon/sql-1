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
 *   Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package com.amazon.opendistroforelasticsearch.sql.ppl;

import static com.amazon.opendistroforelasticsearch.sql.legacy.TestsConstants.TEST_INDEX_ACCOUNT;
import static com.amazon.opendistroforelasticsearch.sql.util.MatcherUtils.rows;
import static com.amazon.opendistroforelasticsearch.sql.util.MatcherUtils.verifyDataRows;

import java.io.IOException;
import org.json.JSONObject;
import org.junit.After;
import org.junit.Before;
import org.junit.jupiter.api.Test;

public class HeadCommandIT extends PPLIntegTestCase {

  @Before
  public void beforeTest() throws IOException {
    setQuerySizeLimit(200);
  }

  @After
  public void afterTest() throws IOException {
    resetQuerySizeLimit();
  }

  @Override
  public void init() throws IOException {
    loadIndex(Index.ACCOUNT);
  }

  @Test
  public void testHead() throws IOException {
    JSONObject result =
        executeQuery(String.format("source=%s | fields firstname, age | head", TEST_INDEX_ACCOUNT));
    verifyDataRows(result,
        rows("Amber", 32),
        rows("Hattie", 36),
        rows("Nanette", 28),
        rows("Dale", 33),
        rows("Elinor", 36),
        rows("Virginia", 39),
        rows("Dillard", 34),
        rows("Mcgee", 39),
        rows("Aurelia", 37),
        rows("Fulton", 23));
  }

  @Test
  public void testHeadWithNumber() throws IOException {
    JSONObject result =
        executeQuery(String.format("source=%s | fields firstname, age | head 3", TEST_INDEX_ACCOUNT));
    verifyDataRows(result,
        rows("Amber", 32),
        rows("Hattie", 36),
        rows("Nanette", 28));
  }
}
