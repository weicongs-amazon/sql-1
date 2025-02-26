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
 *    Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License").
 *    You may not use this file except in compliance with the License.
 *    A copy of the License is located at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    or in the "license" file accompanying this file. This file is distributed
 *    on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *    express or implied. See the License for the specific language governing
 *    permissions and limitations under the License.
 *
 */

package com.amazon.opendistroforelasticsearch.sql.sql;

import static com.amazon.opendistroforelasticsearch.sql.legacy.TestsConstants.TEST_INDEX_BANK;
import static com.amazon.opendistroforelasticsearch.sql.legacy.plugin.RestSqlStatsAction.STATS_API_ENDPOINT;

import com.amazon.opendistroforelasticsearch.sql.legacy.SQLIntegTestCase;
import com.amazon.opendistroforelasticsearch.sql.legacy.metrics.MetricName;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import org.json.JSONObject;
import org.junit.Assert;
import org.junit.Test;
import org.opensearch.client.Request;
import org.opensearch.client.Response;

public class MetricsIT extends SQLIntegTestCase {

  @Override
  protected void init() throws Exception {
    loadIndex(Index.BANK);
  }

  @Test
  public void requestCount() throws IOException, InterruptedException {
    int beforeQueries = requestTotal();
    executeQuery(String.format(Locale.ROOT, "select age from %s", TEST_INDEX_BANK));
    TimeUnit.SECONDS.sleep(2L);

    assertEquals(beforeQueries + 1, requestTotal());
  }

  private Request makeStatRequest() {
    return new Request(
        "GET", STATS_API_ENDPOINT
    );
  }

  private int requestTotal() throws IOException {
    JSONObject jsonObject = new JSONObject(executeStatRequest(makeStatRequest()));
    return jsonObject.getInt(MetricName.REQ_TOTAL.getName());
  }

  private String executeStatRequest(final Request request) throws IOException {
    Response response = client().performRequest(request);
    Assert.assertEquals(200, response.getStatusLine().getStatusCode());

    InputStream is = response.getEntity().getContent();
    StringBuilder sb = new StringBuilder();
    try (BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
      String line;
      while ((line = br.readLine()) != null) {
        sb.append(line);
      }
    }
    return sb.toString();
  }
}
