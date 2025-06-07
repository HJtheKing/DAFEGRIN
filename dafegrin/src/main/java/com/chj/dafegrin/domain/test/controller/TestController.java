package com.chj.dafegrin.domain.test.controller;

import com.chj.dafegrin.domain.bert.service.BertService;
import com.chj.dafegrin.domain.dailyindex.service.DailyIndexService;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDate;
import java.time.ZoneId;

@RestController
@RequiredArgsConstructor
@RequestMapping("/dfg/test")
public class TestController {
    private final DailyIndexService dailyIndexService;
    private final BertService bertService;

    @Value("${naver.news-keyword}")
    private String KEYWORD;

    @Value("${bert.url}")
    private String URL;

    @GetMapping("/bert")
    public ResponseEntity<?> asdf() {
        bertService.getBert(KEYWORD, URL);
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @GetMapping("/kospi")
    public ResponseEntity<?> getKospi() {
        Double kospi = dailyIndexService.getKospi();
        LocalDate now = LocalDate.now(ZoneId.of("Asia/Seoul"));
        dailyIndexService.update(now, kospi);
        return new ResponseEntity<>(HttpStatus.OK);
    }

}
