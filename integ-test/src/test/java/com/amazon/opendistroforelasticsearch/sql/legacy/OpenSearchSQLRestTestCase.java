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
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    or in the "license" file accompanying this file. This file is distributed
 *    on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *    express or implied. See the License for the specific language governing
 *    permissions and limitations under the License.
 *
 */

package com.amazon.opendistroforelasticsearch.sql.legacy;

import java.io.IOException;
import java.util.Map;
import java.util.Optional;
import org.apache.http.Header;
import org.apache.http.HttpHost;
import org.apache.http.auth.AuthScope;
import org.apache.http.auth.UsernamePasswordCredentials;
import org.apache.http.client.CredentialsProvider;
import org.apache.http.conn.ssl.NoopHostnameVerifier;
import org.apache.http.impl.client.BasicCredentialsProvider;
import org.apache.http.message.BasicHeader;
import org.apache.http.ssl.SSLContextBuilder;
import org.apache.http.util.EntityUtils;
import org.json.JSONArray;
import org.json.JSONObject;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.RestClient;
import org.opensearch.client.RestClientBuilder;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.test.rest.OpenSearchRestTestCase;

/**
 * OpenSearch SQL integration test base class to support both security disabled and enabled OpenSearch cluster.
 */
public abstract class OpenSearchSQLRestTestCase extends OpenSearchRestTestCase {

  protected boolean isHttps() {
    boolean isHttps = Optional.ofNullable(System.getProperty("https"))
        .map("true"::equalsIgnoreCase).orElse(false);
    if (isHttps) {
      //currently only external cluster is supported for security enabled testing
      if (!Optional.ofNullable(System.getProperty("tests.rest.cluster")).isPresent()) {
        throw new RuntimeException(
            "external cluster url should be provided for security enabled testing");
      }
    }

    return isHttps;
  }

  protected String getProtocol() {
    return isHttps() ? "https" : "http";
  }

  protected RestClient buildClient(Settings settings, HttpHost[] hosts) throws IOException {
    RestClientBuilder builder = RestClient.builder(hosts);
    if (isHttps()) {
      configureHttpsClient(builder, settings);
    } else {
      configureClient(builder, settings);
    }

    builder.setStrictDeprecationMode(false);
    return builder.build();
  }

  protected static void wipeAllOpenSearchIndices() throws IOException {
    // include all the indices, included hidden indices.
    // https://www.elastic.co/guide/en/elasticsearch/reference/current/cat-indices.html#cat-indices-api-query-params
    Response response = client().performRequest(new Request("GET", "/_cat/indices?format=json&expand_wildcards=all"));
    JSONArray jsonArray = new JSONArray(EntityUtils.toString(response.getEntity(), "UTF-8"));
    for (Object object : jsonArray) {
      JSONObject jsonObject = (JSONObject) object;
      String indexName = jsonObject.getString("index");
      //.opendistro_security isn't allowed to delete from cluster
      if (!indexName.startsWith(".opensearch_dashboards") && !indexName.startsWith(".opendistro")) {
        client().performRequest(new Request("DELETE", "/" + indexName));
      }
    }
  }

  protected static void configureHttpsClient(RestClientBuilder builder, Settings settings)
      throws IOException {
    Map<String, String> headers = ThreadContext.buildDefaultHeaders(settings);
    Header[] defaultHeaders = new Header[headers.size()];
    int i = 0;
    for (Map.Entry<String, String> entry : headers.entrySet()) {
      defaultHeaders[i++] = new BasicHeader(entry.getKey(), entry.getValue());
    }
    builder.setDefaultHeaders(defaultHeaders);
    builder.setHttpClientConfigCallback(httpClientBuilder -> {
      String userName = Optional.ofNullable(System.getProperty("user"))
          .orElseThrow(() -> new RuntimeException("user name is missing"));
      String password = Optional.ofNullable(System.getProperty("password"))
          .orElseThrow(() -> new RuntimeException("password is missing"));
      CredentialsProvider credentialsProvider = new BasicCredentialsProvider();
      credentialsProvider
          .setCredentials(AuthScope.ANY, new UsernamePasswordCredentials(userName, password));
      try {
        return httpClientBuilder.setDefaultCredentialsProvider(credentialsProvider)
            //disable the certificate since our testing cluster just uses the default security configuration
            .setSSLHostnameVerifier(NoopHostnameVerifier.INSTANCE)
            .setSSLContext(SSLContextBuilder.create()
                .loadTrustMaterial(null, (chains, authType) -> true)
                .build());
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    });

    final String socketTimeoutString = settings.get(CLIENT_SOCKET_TIMEOUT);
    final TimeValue socketTimeout =
        TimeValue.parseTimeValue(socketTimeoutString == null ? "60s" : socketTimeoutString,
            CLIENT_SOCKET_TIMEOUT);
    builder.setRequestConfigCallback(
        conf -> conf.setSocketTimeout(Math.toIntExact(socketTimeout.getMillis())));
    if (settings.hasValue(CLIENT_PATH_PREFIX)) {
      builder.setPathPrefix(settings.get(CLIENT_PATH_PREFIX));
    }
  }
}
