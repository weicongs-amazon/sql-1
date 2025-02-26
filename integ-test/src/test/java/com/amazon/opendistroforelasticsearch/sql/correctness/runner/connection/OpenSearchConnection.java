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

package com.amazon.opendistroforelasticsearch.sql.correctness.runner.connection;

import com.amazon.opendistroforelasticsearch.sql.correctness.runner.resultset.DBResult;
import java.io.IOException;
import java.util.List;
import java.util.Properties;
import org.json.JSONObject;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.RestClient;

/**
 * OpenSearch database connection for insertion. This class wraps JDBCConnection to delegate query method.
 */
public class OpenSearchConnection implements DBConnection {

  /**
   * Connection via our OpenSearch JDBC driver
   */
  private final DBConnection connection;

  /**
   * Native OpenSearch REST client for operation unsupported by driver such as CREATE/INSERT
   */
  private final RestClient client;

  public OpenSearchConnection(String connectionUrl, RestClient client) {
    this.connection = new JDBCConnection("OpenSearch", connectionUrl, populateProperties());
    this.client = client;
  }

  @Override
  public String getDatabaseName() {
    return "OpenSearch";
  }

  @Override
  public void connect() {
    connection.connect();
  }

  @Override
  public void create(String tableName, String schema) {
    Request request = new Request("PUT", "/" + tableName);
    request.setJsonEntity(schema);
    performRequest(request);
  }

  @Override
  public void drop(String tableName) {
    performRequest(new Request("DELETE", "/" + tableName));
  }

  @Override
  public void insert(String tableName, String[] columnNames, List<Object[]> batch) {
    Request request = new Request("POST", "/" + tableName + "/_bulk?refresh=true");
    request.setJsonEntity(buildBulkBody(columnNames, batch));
    performRequest(request);
  }

  @Override
  public DBResult select(String query) {
    return connection.select(query);
  }

  @Override
  public void close() {
    // Only close database connection and leave OpenSearch REST connection alone
    // because it's initialized and manged by OpenSearch test base class.
    connection.close();
  }

  private Properties populateProperties() {
    Properties properties = new Properties();
    if (Boolean.parseBoolean(System.getProperty("https", "false"))) {
      properties.put("useSSL", "true");
    }
    if (!System.getProperty("user", "").isEmpty()) {
      properties.put("user", System.getProperty("user"));
      properties.put("password", System.getProperty("password", ""));
      properties.put("trustSelfSigned", "true");
      properties.put("hostnameVerification", "false");
    }
    return properties;
  }

  private void performRequest(Request request) {
    try {
      Response response = client.performRequest(request);
      int status = response.getStatusLine().getStatusCode();
      if (status != 200) {
        throw new IllegalStateException("Failed to perform request. Error code: " + status);
      }
    } catch (IOException e) {
      throw new IllegalStateException("Failed to perform request", e);
    }
  }

  private String buildBulkBody(String[] columnNames, List<Object[]> batch) {
    StringBuilder body = new StringBuilder();
    for (Object[] fieldValues : batch) {
      JSONObject json = new JSONObject();
      for (int i = 0; i < columnNames.length; i++) {
        if (fieldValues[i] != null) {
          json.put(columnNames[i], fieldValues[i]);
        }
      }

      body.append("{\"index\":{}}\n").
          append(json).append("\n");
    }
    return body.toString();
  }

}
